# 光线追踪
| 202311081051 | 王婧怡 | 计算机科学与技术 |
| --- | --- | --- |


## <font style="color:rgb(15, 17, 21);">一、实验目标</font>
1. <font style="color:rgb(15, 17, 21);">理论理解：深入理解光线投射（Ray Casting）与光线追踪（Ray Tracing）的本质区别</font>
2. <font style="color:rgb(15, 17, 21);">全局光照：掌握通过发射次级射线实现硬阴影和理想镜面反射的方法</font>
3. <font style="color:rgb(15, 17, 21);">GPU编程思维：学习将递归光线追踪算法改写为适合GPU并行计算的迭代模式</font>

## <font style="color:rgb(15, 17, 21);">二、实验原理</font>
<font style="color:rgb(15, 17, 21);">本实验采用经典的Whitted-Style光线追踪模型，其核心思想是模拟光线在场景中的完整传播路径：</font>

### <font style="color:rgb(15, 17, 21);">2.1 光线追踪流程</font>
<font style="color:rgb(15, 17, 21);">当主光线从摄像机出发击中物体表面时：</font>

1. <font style="color:rgb(15, 17, 21);">阴影测试：从交点向光源发射阴影射线，判断该点是否被遮挡</font>
2. <font style="color:rgb(15, 17, 21);">材质分支处理</font><font style="color:rgb(15, 17, 21);">：</font>
    - <font style="color:rgb(15, 17, 21);">漫反射材质</font><font style="color:rgb(15, 17, 21);">：按Phong模型计算颜色，终止光线传播</font>
    - <font style="color:rgb(15, 17, 21);">镜面材质：根据反射定律计算反射方向，生成新的反射射线继续追踪</font>

### <font style="color:rgb(15, 17, 21);">2.2 关键技术原理</font>
<font style="color:rgb(15, 17, 21);">反射向量计算： 

$$
R=L_{in}−2(L_{in}⋅N)N 
$$

<font style="color:rgb(15, 17, 21);">其中，

$$
L_{in} 
$$

为入射光线方向，N为表面法向量。</font>

<font style="color:rgb(15, 17, 21);">能量衰减机制</font><font style="color:rgb(15, 17, 21);">：</font>

+ <font style="color:rgb(15, 17, 21);">每次镜面反射时，光线能量乘以反射率（本实验设为0.8）</font>
+ <font style="color:rgb(15, 17, 21);">通过throughput变量累积衰减系数</font>
+ <font style="color:rgb(15, 17, 21);">最终颜色 = 漫反射颜色 × 累积throughput</font>

## <font style="color:rgb(15, 17, 21);">三、实验任务实现</font>
### <font style="color:rgb(15, 17, 21);">3.1 场景搭建（任务1）</font>
**<font style="color:rgb(15, 17, 21);">几何体定义</font>**<font style="color:rgb(15, 17, 21);">：</font>

1. <font style="color:rgb(15, 17, 21);">无限大平面（地板）</font>
    - <font style="color:rgb(15, 17, 21);">位置：y = -1.0</font>
    - <font style="color:rgb(15, 17, 21);">法线：(0, 1, 0)</font>
    - <font style="color:rgb(15, 17, 21);">材质：漫反射，黑白棋盘格纹理</font>
    - <font style="color:rgb(15, 17, 21);">纹理生成：通过交点x和z坐标的奇偶性判断</font>
2. <font style="color:rgb(15, 17, 21);">红色漫反射球</font>
    - <font style="color:rgb(15, 17, 21);">位置：(-1.2, 0.0, 0.0)</font>
    - <font style="color:rgb(15, 17, 21);">半径：1.0</font>
    - <font style="color:rgb(15, 17, 21);">材质：漫反射</font>
3. <font style="color:rgb(15, 17, 21);">银色镜面球</font>
    - <font style="color:rgb(15, 17, 21);">位置：(1.2, 0.0, 0.0)</font>
    - <font style="color:rgb(15, 17, 21);">半径：1.0</font>
    - <font style="color:rgb(15, 17, 21);">材质：纯镜面反射</font>

**<font style="color:rgb(15, 17, 21);">材质ID系统</font>**<font style="color:rgb(15, 17, 21);">：</font>

```plain
MAT_DIFFUSE = 0  # 漫反射材质
MAT_MIRROR = 1   # 镜面反射材质
```

### <font style="color:rgb(15, 17, 21);">3.2 迭代式光线追踪（任务2）</font>
**<font style="color:rgb(15, 17, 21);">核心循环结构</font>**<font style="color:rgb(15, 17, 21);">：</font>

```plain
for bounce in range(max_bounces[None]):
    # 场景求交
    t, N, obj_color, mat_id = scene_intersect(ro, rd)
    
    if mat_id == MAT_MIRROR:
        # 镜面反射：更新光线方向和起点
        ro = p + N * 1e-4
        rd = normalize(reflect(rd, N))
        throughput *= 0.8 * obj_color
        # 继续循环
    elif mat_id == MAT_DIFFUSE:
        # 漫反射：计算颜色并终止
        final_color += throughput * direct_light
        break
```

**<font style="color:rgb(15, 17, 21);">光线吞吐量管理</font>**<font style="color:rgb(15, 17, 21);">：</font>

+ <font style="color:rgb(15, 17, 21);">初始throughput = (1.0, 1.0, 1.0)</font>
+ <font style="color:rgb(15, 17, 21);">每次镜面反射乘以反射率0.8</font>
+ <font style="color:rgb(15, 17, 21);">最终颜色累加时乘以累积throughput</font>

### <font style="color:rgb(15, 17, 21);">3.3 硬阴影与精度优化（任务3）</font>
**<font style="color:rgb(15, 17, 21);">阴影检测实现</font>**<font style="color:rgb(15, 17, 21);">：</font>

```plain
# 发射阴影射线
shadow_ray_orig = p + N * 1e-4  # 法线偏移防止自相交
shadow_t, _, _, _ = scene_intersect(shadow_ray_orig, L)
dist_to_light = (light_pos - p).norm()

# 判断遮挡
if shadow_t < dist_to_light:
    in_shadow = 1.0  # 被遮挡
```

**<font style="color:rgb(15, 17, 21);">关键问题解决</font>**<font style="color:rgb(15, 17, 21);">：</font>

+ <font style="color:rgb(15, 17, 21);">Shadow Acne问题：射线起点沿法线偏移1e-4</font>
+ <font style="color:rgb(15, 17, 21);">自相交Bug：同样在反射射线起点应用偏移</font>
+ <font style="color:rgb(15, 17, 21);">偏移量选择：1e-4在保证精度的同时避免可见偏移</font>

### <font style="color:rgb(15, 17, 21);">3.4 UI交互面板（任务4）</font>
**<font style="color:rgb(15, 17, 21);">控制参数</font>**<font style="color:rgb(15, 17, 21);">：</font>

1. <font style="color:rgb(15, 17, 21);">光源位置：Light X/Y/Z，范围-5~5</font>
2. <font style="color:rgb(15, 17, 21);">最大弹射次数：Max Bounces，范围1~5</font>

**<font style="color:rgb(15, 17, 21);">交互效果</font>**<font style="color:rgb(15, 17, 21);">：</font>

+ <font style="color:rgb(15, 17, 21);">光源移动时阴影实时变化</font>
+ <font style="color:rgb(15, 17, 21);">弹射次数=1时无反射效果</font>
+ <font style="color:rgb(15, 17, 21);">弹射次数>1时出现镜面反射和"镜中世界"</font>

## <font style="color:rgb(15, 17, 21);">四、实验结果与分析</font>
### <font style="color:rgb(15, 17, 21);">4.1 关键技术对比</font>
| <font style="color:rgb(15, 17, 21);">特性</font> | <font style="color:rgb(15, 17, 21);">光线投射</font> | <font style="color:rgb(15, 17, 21);">光线追踪</font> |
| --- | --- | --- |
| <font style="color:rgb(15, 17, 21);">光线路数</font> | <font style="color:rgb(15, 17, 21);">单一路径</font> | <font style="color:rgb(15, 17, 21);">多路径（递归）</font> |
| <font style="color:rgb(15, 17, 21);">反射效果</font> | <font style="color:rgb(15, 17, 21);">无</font> | <font style="color:rgb(15, 17, 21);">有（镜面反射）</font> |
| <font style="color:rgb(15, 17, 21);">阴影实现</font> | <font style="color:rgb(15, 17, 21);">简单</font> | <font style="color:rgb(15, 17, 21);">精确硬阴影</font> |
| <font style="color:rgb(15, 17, 21);">计算复杂度</font> | <font style="color:rgb(15, 17, 21);">O(n)</font> | <font style="color:rgb(15, 17, 21);">O(n^bounces)</font> |


### <font style="color:rgb(15, 17, 21);">4.2 实验观察</font>
**<font style="color:rgb(15, 17, 21);">硬阴影效果</font>**<font style="color:rgb(15, 17, 21);">：</font>

+ <font style="color:rgb(15, 17, 21);">光源位置变化时阴影形状和方向实时更新</font>
+ <font style="color:rgb(15, 17, 21);">阴影边界清晰，符合点光源特性</font>

**<font style="color:rgb(15, 17, 21);">镜面反射效果</font>**<font style="color:rgb(15, 17, 21);">：</font>

+ <font style="color:rgb(15, 17, 21);">镜面球体清晰反射红色球体和棋盘地板</font>
+ <font style="color:rgb(15, 17, 21);">反射次数增加时，"镜中世界"的细节逐渐丰富</font>
+ <font style="color:rgb(15, 17, 21);">能量衰减使远离光源的反射逐渐变暗</font>

**<font style="color:rgb(15, 17, 21);">浮点数精度处理</font>**<font style="color:rgb(15, 17, 21);">：</font>

+ <font style="color:rgb(15, 17, 21);">未添加偏移时出现大量黑色噪点</font>
+ <font style="color:rgb(15, 17, 21);">添加1e-4偏移后画面干净清晰</font>

下列是分别调整XYZ轴的光源位置以及最大弹射次数后的效果
<img width="804" height="647" alt="x轴" src="https://github.com/user-attachments/assets/fd5b416e-525e-4c96-bd0c-cef752cd3d11" />
<img width="804" height="647" alt="Y轴" src="https://github.com/user-attachments/assets/9a74eb70-66db-40f8-8f95-ee8ba7ca761b" />
<img width="804" height="647" alt="Z轴" src="https://github.com/user-attachments/assets/55df782d-9240-423f-8177-f8fe98b3c5a8" />
<img width="804" height="647" alt="Max" src="https://github.com/user-attachments/assets/65095a35-8204-4287-9dff-3e8cfde15f71" />
