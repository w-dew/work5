import taichi as ti

# 初始化 Taichi GPU 后端 (Mac 自动调用 Metal，Win 调用 CUDA/Vulkan)
ti.init(arch=ti.gpu)

res_x, res_y = 800, 600
pixels = ti.Vector.field(3, dtype=ti.f32, shape=(res_x, res_y))

# 交互参数
light_pos_x = ti.field(ti.f32, shape=())
light_pos_y = ti.field(ti.f32, shape=())
light_pos_z = ti.field(ti.f32, shape=())
max_bounces = ti.field(ti.i32, shape=())
glass_ior = ti.field(ti.f32, shape=())      # 玻璃折射率
spp = ti.field(ti.i32, shape=())            # 每像素采样数 (MSAA)

# 材质常量枚举
MAT_DIFFUSE = 0
MAT_MIRROR = 1
MAT_GLASS = 2   # 新增：玻璃（折射）材质

@ti.func
def normalize(v):
    return v / v.norm(1e-5)

@ti.func
def reflect(I, N):
    return I - 2.0 * I.dot(N) * N

@ti.func
def refract(I, N, eta):
    """
    斯涅尔定律折射。I 为入射方向（指向物体表面，已归一化），
    N 为已经按入射/射出朝向修正过的法线，eta = n1/n2。
    返回 (折射方向, 是否发生全反射)
    """
    cosi = -I.dot(N)
    k = 1.0 - eta * eta * (1.0 - cosi * cosi)
    out_dir = ti.Vector([0.0, 0.0, 0.0])
    total_internal_reflection = False
    if k < 0.0:
        # 全反射：折射分量消失，只能反射
        total_internal_reflection = True
    else:
        out_dir = eta * I + (eta * cosi - ti.sqrt(k)) * N
    return out_dir, total_internal_reflection

@ti.func
def fresnel_schlick(cosine, ior):
    """Schlick 近似计算反射率（菲涅尔反射）"""
    r0 = (1.0 - ior) / (1.0 + ior)
    r0 = r0 * r0
    return r0 + (1.0 - r0) * ti.pow(1.0 - cosine, 5.0)

@ti.func
def intersect_sphere(ro, rd, center, radius):
    """球体求交，返回 (距离 t, 法线 normal)"""
    t = -1.0
    normal = ti.Vector([0.0, 0.0, 0.0])
    oc = ro - center
    b = 2.0 * oc.dot(rd)
    c = oc.dot(oc) - radius * radius
    delta = b * b - 4.0 * c
    if delta > 0:
        t1 = (-b - ti.sqrt(delta)) / 2.0
        t2 = (-b + ti.sqrt(delta)) / 2.0
        # 注意：玻璃球内部出射时，光线起点在球内，最近的有效交点
        # 可能是较大的根 t2（从内部射向球面）。这里优先取大于 1e-4 的最小正根。
        if t1 > 1e-4:
            t = t1
            p = ro + rd * t
            normal = normalize(p - center)
        elif t2 > 1e-4:
            t = t2
            p = ro + rd * t
            normal = normalize(p - center)
    return t, normal

@ti.func
def intersect_plane(ro, rd, plane_y):
    """水平无限大平面求交"""
    t = -1.0
    normal = ti.Vector([0.0, 1.0, 0.0]) # 法线永远朝上
    if ti.abs(rd.y) > 1e-5:
        t1 = (plane_y - ro.y) / rd.y
        if t1 > 0:
            t = t1
    return t, normal

@ti.func
def scene_intersect(ro, rd):
    """
    遍历场景，寻找最近交点。
    返回: (t, 法线 N, 颜色 color, 材质 mat_id)
    """
    min_t = 1e10
    hit_n = ti.Vector([0.0, 0.0, 0.0])
    hit_c = ti.Vector([0.0, 0.0, 0.0])
    hit_mat = MAT_DIFFUSE

    # 1. 检测玻璃球（原红色漫反射球，现改为玻璃材质）
    t, n = intersect_sphere(ro, rd, ti.Vector([-1.2, 0.0, 0.0]), 1.0)
    if 0 < t < min_t:
        min_t = t
        hit_n = n
        hit_c = ti.Vector([1.0, 1.0, 1.0])  # 玻璃基本不吸收颜色（近似无色透明）
        hit_mat = MAT_GLASS

    # 2. 检测银色镜面球
    t, n = intersect_sphere(ro, rd, ti.Vector([1.2, 0.0, 0.0]), 1.0)
    if 0 < t < min_t:
        min_t = t
        hit_n = n
        hit_c = ti.Vector([0.9, 0.9, 0.9]) # 镜面反射基础色
        hit_mat = MAT_MIRROR

    # 3. 检测地板
    t, n = intersect_plane(ro, rd, -1.0)
    if 0 < t < min_t:
        min_t = t
        hit_n = n
        hit_mat = MAT_DIFFUSE
        # 生成棋盘格纹理
        p = ro + rd * t
        grid_scale = 2.0
        ix = ti.floor(p.x * grid_scale)
        iz = ti.floor(p.z * grid_scale)
        # 判断坐标的奇偶性来交替颜色
        if (ix + iz) % 2 == 0:
            hit_c = ti.Vector([0.3, 0.3, 0.3]) # 灰色格子
        else:
            hit_c = ti.Vector([0.8, 0.8, 0.8]) # 白色格子

    return min_t, hit_n, hit_c, hit_mat

@ti.func
def trace_ray(ro, rd, bg_color, light_pos):
    """追踪单条主射线（含其折射/反射子路径），返回最终颜色。
    折射/反射使用俄罗斯轮盘式的随机选择（按菲涅尔反射率加权），
    配合外层的多重采样（MSAA）求平均，从统计上正确地还原玻璃的
    半透半反效果，而无需真正的递归分支。"""
    final_color = ti.Vector([0.0, 0.0, 0.0])
    throughput = ti.Vector([1.0, 1.0, 1.0])

    for bounce in range(max_bounces[None]):
        t, N, obj_color, mat_id = scene_intersect(ro, rd)

        # 如果没击中任何物体，加上背景色并结束追踪
        if t > 1e9:
            final_color += throughput * bg_color
            break

        p = ro + rd * t

        # 分支 1：镜面反射材质
        if mat_id == MAT_MIRROR:
            ro = p + N * 1e-4
            rd = normalize(reflect(rd, N))
            throughput *= 0.8 * obj_color
            # 不跳出循环，继续追踪反射射线

        # 分支 2：玻璃（折射）材质
        elif mat_id == MAT_GLASS:
            ior = glass_ior[None]
            n_unit = normalize(N)
            cosi = ti.math.clamp(rd.dot(n_unit), -1.0, 1.0)

            etai = 1.0
            etat = ior
            oriented_n = n_unit
            if cosi > 0.0:
                # 光线从物体内部射出 -> 表面
                oriented_n = -n_unit
                etai, etat = etat, etai
            else:
                cosi = -cosi

            eta = etai / etat
            refr_dir, tir = refract(rd, oriented_n, eta)

            reflect_prob = 1.0  # 默认（发生全反射时）必为反射
            if not tir:
                reflect_prob = fresnel_schlick(cosi, ior)

            # 用菲涅尔反射率作为随机选择反射/折射的概率（俄罗斯轮盘）
            if (not tir) and ti.random() > reflect_prob:
                # 折射：进入或离开玻璃
                ro = p - oriented_n * 1e-4
                rd = normalize(refr_dir)
                throughput *= obj_color  # 近似无色透明，不额外衰减
            else:
                # 反射（含全反射的情况）
                ro = p + oriented_n * 1e-4
                rd = normalize(reflect(rd, oriented_n))
                throughput *= obj_color
            # 不跳出循环，继续追踪反射/折射后的射线

        # 分支 3：漫反射材质
        elif mat_id == MAT_DIFFUSE:
            L = normalize(light_pos - p)

            # --- 硬阴影检测 ---
            shadow_ray_orig = p + N * 1e-4
            shadow_t, _, _, _ = scene_intersect(shadow_ray_orig, L)

            dist_to_light = (light_pos - p).norm()
            in_shadow = 0.0
            if shadow_t < dist_to_light:
                in_shadow = 1.0

            ambient = 0.2 * obj_color
            direct_light = ambient

            if in_shadow == 0.0:
                diff = ti.max(0.0, N.dot(L))
                diffuse = 0.8 * diff * obj_color
                direct_light += diffuse

            final_color += throughput * direct_light
            # 漫反射表面会打散光线，主射线到此终止
            break

    return final_color

@ti.kernel
def render():
    light_pos = ti.Vector([light_pos_x[None], light_pos_y[None], light_pos_z[None]])
    bg_color = ti.Vector([0.05, 0.15, 0.2])

    for i, j in pixels:
        color_sum = ti.Vector([0.0, 0.0, 0.0])
        n_samples = spp[None]

        # --- MSAA: 每个像素内随机采样多次，平均颜色以消除锯齿 ---
        for s in range(n_samples):
            # 在像素内 [-0.5, 0.5) 范围随机抖动采样点
            jitter_x = ti.random() - 0.5
            jitter_y = ti.random() - 0.5

            u = (i + jitter_x - res_x / 2.0) / res_y * 2.0
            v = (j + jitter_y - res_y / 2.0) / res_y * 2.0

            ro = ti.Vector([0.0, 1.0, 5.0])  # 摄像机稍微抬高一点
            rd = normalize(ti.Vector([u, v - 0.2, -1.0]))  # 视角微微向下看

            color_sum += trace_ray(ro, rd, bg_color, light_pos)

        final_color = color_sum / n_samples

        # 写入像素并进行色调映射
        pixels[i, j] = ti.math.clamp(final_color, 0.0, 1.0)

def main():
    window = ti.ui.Window("Ray Tracing Demo", (res_x, res_y))
    canvas = window.get_canvas()
    gui = window.get_gui()

    # 初始化光源位置和弹射次数
    light_pos_x[None] = 2.0
    light_pos_y[None] = 4.0
    light_pos_z[None] = 3.0
    max_bounces[None] = 5      # 玻璃折射+反射需要更多弹射次数才能收敛
    glass_ior[None] = 1.5      # 普通玻璃折射率约 1.5
    spp[None] = 4               # 每像素采样数（MSAA），越大越平滑但越慢

    while window.running:
        render()
        canvas.set_image(pixels)

        with gui.sub_window("Controls", 0.72, 0.05, 0.26, 0.30):
            light_pos_x[None] = gui.slider_float('Light X', light_pos_x[None], -5.0, 5.0)
            light_pos_y[None] = gui.slider_float('Light Y', light_pos_y[None], 1.0, 8.0)
            light_pos_z[None] = gui.slider_float('Light Z', light_pos_z[None], -5.0, 5.0)
            max_bounces[None] = gui.slider_int('Max Bounces', max_bounces[None], 1, 8)
            glass_ior[None] = gui.slider_float('Glass IOR', glass_ior[None], 1.0, 2.5)
            spp[None] = gui.slider_int('MSAA Samples', spp[None], 1, 16)

        window.show()

if __name__ == '__main__':
    main()
