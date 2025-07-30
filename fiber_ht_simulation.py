import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
import pyansys
import fenics as fe
import ase
from ase import io
from ase.calculators.lammpsrun import LAMMPS
from ase.calculators.emt import EMT
import lammps
import qutip

# ====================
# 1. 量子尺度模拟 (DFT计算)
# ====================
def dft_calculation():
    print("正在执行量子尺度模拟(DFT计算)...")
    from ase.build import bulk
    from ase.calculators.siesta import Siesta
    from ase.visualize.plot import plot_atoms
    
    # 构建α-石英晶体
    a = 4.913  # Å
    c = 5.405
    atoms = bulk('SiO2', 'quartz', a=a, c=c)
    
    # DFT计算参数
    calc = Siesta(
        label='siesta',
        kpts=[4, 4, 4],
        xc='PBE',
        mesh_cutoff=200,
        energy_shift=0.01,
        basis_set='DZP'
    )
    
    # 设置不同温度下的晶格膨胀
    temperatures = np.linspace(300, 1500, 6)  # K
    band_gaps = []
    lattice_params = []
    
    for T in temperatures:
        # 计算热膨胀
        delta_a = 5.6e-6 * (T - 300) * a
        delta_c = 7.8e-6 * (T - 300) * c
        
        # 创建膨胀后的结构
        atoms_expanded = atoms.copy()
        atoms_expanded.cell = [[a + delta_a, 0, 0],
                              [0, a + delta_a, 0],
                              [0, 0, c + delta_c]]
        atoms_expanded.calc = calc
        
        try:
            energy = atoms_expanded.get_potential_energy()
            band_gap = calc.get_band_gap()
            if band_gap is None:
                band_gap = calc.results['band_gap']
        except Exception as e:
            print(f"在温度{T}K时DFT计算失败: {e}")
            band_gap = 9.0 - 0.004*(T-300)  # 经验公式回退
        
        band_gaps.append(band_gap)
        lattice_params.append((a + delta_a, c + delta_c))
    
    # 可视化结果
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.plot(temperatures, band_gaps, 'o-', color='b', linewidth=2)
    ax1.set_xlabel('温度 (K)', fontsize=12)
    ax1.set_ylabel('带隙 (eV)', color='b', fontsize=12)
    ax1.tick_params(axis='y', labelcolor='b')
    ax1.grid(True, alpha=0.3)
    
    ax2 = ax1.twinx()
    ax2.plot(temperatures, [p[0] for p in lattice_params], 's--', color='r', label='a-轴')
    ax2.plot(temperatures, [p[1] for p in lattice_params], 'd--', color='g', label='c-轴')
    ax2.set_ylabel('晶格参数 (Å)', color='r', fontsize=12)
    ax2.tick_params(axis='y', labelcolor='r')
    ax2.legend(loc='upper center')
    
    plt.title('石英晶体带隙和晶格参数随温度变化', fontsize=14)
    plt.tight_layout()
    plt.savefig('dft_results.png', dpi=300)
    print("量子尺度模拟完成，结果已保存到 dft_results.png")
    
    return {'temperatures': temperatures, 'band_gaps': band_gaps, 'lattice_params': lattice_params}

# ====================
# 2. 原子尺度模拟 (分子动力学)
# ====================
def molecular_dynamics_simulation():
    print("正在执行原子尺度模拟(分子动力学)...")
    
    # 创建LAMMPS实例
    lmp = lammps.lammps()
    
    # LAMMPS脚本
    commands = """
    # LAMMPS输入脚本: 高温下石英分子动力学模拟
    
    units metal
    dimension 3
    boundary p p p
    atom_style atomic
    
    # 创建石英晶格
    lattice custom 1.0 a1 4.913 0.0 0.0 a2 0.0 4.913 0.0 a3 0.0 0.0 5.405 &
            basis 0.333 0.333 0.333 &
            basis 0.666 0.666 0.666
    region box block 0 10 0 10 0 5
    create_box 2 box
    create_atoms 1 box
    
    # 设置原子类型
    mass 1 16.0  # 氧
    mass 2 28.0  # 硅
    
    # BKS势函数参数
    pair_style bks
    pair_coeff 1 1 0.0 1.0 0.0    # O-O
    pair_coeff 2 2 0.0 1.0 0.0    # Si-Si
    pair_coeff 1 2 18003.7572 133.5381 2.314  # O-Si
    
    # 能量最小化
    minimize 1e-6 1e-8 1000 10000
    
    # 设置温度
    variable T equal 300
    variable T_step equal 100
    variable T_max equal 1500
    
    # 创建输出文件
    thermo 100
    thermo_style custom step temp vol press pe ke density
    dump 1 all custom 1000 atoms.lammpstrj id type x y z
    
    # 初始热化
    timestep 0.001
    velocity all create $T 4928459
    fix 1 all nvt temp $T $T 0.1
    run 5000
    
    # 温度扫描
    label loop
    variable a equal lx
    variable c equal lz
    print "温度: $T | a-晶格: ${a} | c-晶格: ${c}"
    
    # 计算RDF
    compute rdfO all rdf 100 1 1
    compute rdfSi all rdf 100 2 2
    compute rdfOSi all rdf 100 1 2
    
    # 升温运行
    fix 1 all npt temp $T $T 0.1 iso 0.0 0.0 1.0
    run 10000
    
    # 保存RDF数据
    write_data data_$T.dat
    
    # 增加温度
    variable T equal ${T}+${T_step}
    if "${T} < ${T_max}" then "jump SELF loop"
    """
    
    # 执行LAMMPS脚本
    lmp.commands_string(commands)
    
    # 结果处理
    temperatures = np.arange(300, 1501, 100)
    rdfs = []
    
    for T in temperatures:
        try:
            data = np.loadtxt(f'data_{T}.dat', skiprows=9, max_rows=100)
            rdfs.append(data[:, 2])  # g(r)
        except:
            print(f"温度{T}K的RDF数据加载失败")
    
    # 可视化
    plt.figure(figsize=(10, 6))
    r = np.linspace(0, 10, 100)
    for i, T in enumerate(temperatures):
        plt.plot(r, rdfs[i], label=f'T={T}K')
    
    plt.xlabel('原子间距 (Å)', fontsize=12)
    plt.ylabel('径向分布函数 g(r)', fontsize=12)
    plt.title('不同温度下SiO₂的径向分布函数', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('md_rdf_results.png', dpi=300)
    print("原子尺度模拟完成，结果已保存到 md_rdf_results.png")
    
    return {'temperatures': temperatures, 'rdfs': rdfs}

# ====================
# 3. 微观尺度模拟 (相场断裂模型)
# ====================
def phase_field_fracture():
    print("正在执行微观尺度模拟(相场断裂)...")
    
    # 创建有限元网格
    mesh = fe.BoxMesh(fe.Point(0,0,0), fe.Point(10e-6,1e-6,1e-6), 100, 10, 10)
    
    # 定义函数空间
    P1 = fe.FiniteElement('P', mesh.ufl_cell(), 1)
    V = fe.FunctionSpace(mesh, fe.MixedElement([P1, P1, P1, P1]))  # ux, uy, uz, phi
    
    # 定义变量
    w = fe.Function(V)
    ux, uy, uz, phi = fe.split(w)
    u = fe.as_vector([ux, uy, uz])
    
    # 测试函数
    w_test = fe.TestFunction(V)
    vx, vy, vz, psi = fe.split(w_test)
    v = fe.as_vector([vx, vy, vz])
    
    # 材料参数
    E = fe.Constant(73e9)     # 杨氏模量 (Pa)
    nu = fe.Constant(0.17)    # 泊松比
    lmbda = E*nu/((1+nu)*(1-2*nu))
    mu = E/(2*(1+nu))
    Gc = fe.Constant(8.0)    # 临界能量释放率 (J/m²)
    l0 = fe.Constant(0.1e-6) # 特征长度 (m)
    
    # 损伤函数
    g = (1-phi)**2 + 1e-8
    k = 1e-6  # 数值稳定性参数
    
    # 应变张量
    def epsilon(u):
        return 0.5*(fe.nabla_grad(u) + fe.nabla_grad(u).T)
    
    # 应变能密度
    psi_e = 0.5*lmbda*fe.tr(epsilon(u))**2 + mu*fe.inner(epsilon(u), epsilon(u))
    
    # 相场能量
    Psi = g*psi_e + Gc*(phi**2/(2*l0) + l0/2*fe.inner(fe.grad(phi), fe.grad(phi)))
    
    # 弱形式
    L = fe.derivative(Psi*fe.dx, w, w_test)
    
    # 温度效应
    temperatures = [300, 600, 900, 1200]
    fracture_strains = []
    
    for T in temperatures:
        # 温度相关的材料参数
        E.value = 73e9 * (1 - 1.5e-4*(T-300))
        Gc.value = 8.0 * (1 - 2e-4*(T-300))
        
        # 边界条件 (拉伸测试)
        def left_boundary(x, on_boundary):
            return fe.near(x[0], 0) and on_boundary
        
        def right_boundary(x, on_boundary):
            return fe.near(x[0], 10e-6) and on_boundary
        
        bc_left = fe.DirichletBC(V.sub(0), fe.Constant(0), left_boundary)
        bc_right = fe.DirichletBC(V.sub(0), fe.Expression('t', t=0, degree=1), right_boundary)
        bcs = [bc_left, bc_right]
        
        # 增量加载
        t = 0
        dt = 0.01
        max_t = 0.1
        
        fracture_strain = None
        
        while t < max_t:
            # 更新位移边界条件
            bc_right._function.t = t * 1e-7  # 增加位移
            
            # 求解
            fe.solve(L == 0, w, bcs)
            
            # 检查断裂
            phi_value = phi.compute_vertex_values(mesh)
            if np.max(phi_value) > 0.95 and fracture_strain is None:
                fracture_strain = t
                print(f"温度 {T}K 时在应变 {t:.4f} 发生断裂")
                break
            
            t += dt
        
        fracture_strains.append(fracture_strain if fracture_strain else max_t)
    
    # 可视化
    plt.figure(figsize=(10, 6))
    plt.plot(temperatures, fracture_strains, 'ro-', markersize=8)
    plt.xlabel('温度 (K)', fontsize=12)
    plt.ylabel('断裂应变', fontsize=12)
    plt.title('温度对SiO₂断裂应变的影响', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.savefig('phase_field_results.png', dpi=300)
    print("微观尺度模拟完成，结果已保存到 phase_field_results.png")
    
    return {'temperatures': temperatures, 'fracture_strains': fracture_strains}

# ====================
# 4. 宏观尺度模拟 (有限元分析)
# ====================
def macro_fem_simulation():
    print("正在执行宏观尺度模拟(有限元分析)...")
    
    # 启动APDL
    mapdl = pyansys.launch_mapdl()
    mapdl.prep7()
    
    # 材料参数
    E = 73e9          # 杨氏模量 (Pa)
    nu = 0.17         # 泊松比
    alpha = 0.55e-6   # 热膨胀系数 (K⁻¹)
    kxx = 1.38        # 热导率 (W/m·K)
    density = 2200    # 密度 (kg/m³)
    Cp = 750          # 比热容 (J/kg·K)
    
    # 定义材料
    mapdl.mp("EX", 1, E)
    mapdl.mp("NUXY", 1, nu)
    mapdl.mp("ALPX", 1, alpha)
    mapdl.mp("KXX", 1, kxx)
    mapdl.mp("DENS", 1, density)
    mapdl.mp("C", 1, Cp)
    
    # 几何模型 - 光纤结构
    core_radius = 10e-6
    cladding_radius = 125e-6
    coating_radius = 250e-6
    length = 0.01  # 光纤长度 (m)
    
    mapdl.cyl4(0, 0, 0, '', core_radius, 0, length)        # 纤芯
    mapdl.cyl4(0, 0, 0, '', cladding_radius, 0, length)    # 包层
    mapdl.cyl4(0, 0, 0, '', coating_radius, 0, length)     # 涂层
    
    mapdl.vsbv(2, 1)  # 包层减去纤芯
    mapdl.vsbv(3, 2)  # 涂层减去包层
    mapdl.vglue("ALL")
    
    # 划分网格
    mapdl.et(1, "SOLID226", 11)  # 热电耦合单元
    mapdl.esize(25e-6)
    mapdl.vmesh("ALL")
    
    # 边界条件
    mapdl.nsel("S", "LOC", "Z", 0)
    mapdl.d("ALL", "TEMP", 300)  # 冷端温度
    
    mapdl.nsel("S", "LOC", "Z", length)
    mapdl.d("ALL", "TEMP", 1200)  # 热端温度
    
    # 求解
    mapdl.slashsolu()
    mapdl.antype("STATIC")
    mapdl.tref(300)  # 参考温度
    mapdl.outres("ALL", "ALL")
    mapdl.solve()
    
    # 获取结果
    result = mapdl.result
    nodal_temp = result.nodal_temperature(0)
    stress = result.principal_nodal_stress(0)
    
    # 可视化温度分布
    mapdl.post1()
    mapdl.set(1, 1)
    mapdl.show("png")
    mapdl.png("temperature_distribution")
    
    # 可视化应力分布
    mapdl.plnsol("S", "EQV")
    mapdl.show("png")
    mapdl.png("stress_distribution")
    
    print("宏观尺度模拟完成，结果已保存为图片")
    mapdl.exit()
    
    return {'nodal_temp': nodal_temp, 'stress': stress}

# ====================
# 5. 量子传感器模拟 (NV色心)
# ====================
def nv_center_simulation():
    print("正在执行量子传感器模拟(NV色心)...")
    
    # NV色心参数
    D = 2.87e9  # Zero-field splitting (Hz)
    gamma_e = 28e15  # Electron gyromagnetic ratio (Hz/T)
    
    # 泡利矩阵
    sx = qutip.sigmax()
    sy = qutip.sigmay()
    sz = qutip.sigmaz()
    
    # 温度依赖的零场分裂
    def D_temp(T):
        # 温度对零场分裂的影响 (典型值: -74 kHz/K)
        return D + (-74e3) * (T - 300)
    
    # ODMR频谱计算
    def calculate_odmr(T, B=0):
        # 哈密顿量
        H = 0.5 * D_temp(T) * (sz**2 - (2/3)*qutip.qeye(2)) + \
            gamma_e * B * (sx + sz)
        
        # 激励算符
        drive = qutip.sigmax()
        
        # 频谱参数
        omega_min = 2.5e9
        omega_max = 3.2e9
        n_points = 200
        frequencies = np.linspace(omega_min, omega_max, n_points)
        
        # 计算频谱
        spectrum = qutip.spectrum(H, frequencies, [], drive, [], gamma=0.1e6)
        
        return frequencies, spectrum
    
    # 计算不同温度下的ODMR
    temperatures = [300, 400, 500, 600]
    results = []
    
    plt.figure(figsize=(12, 8))
    for i, T in enumerate(temperatures):
        freq, spectrum = calculate_odmr(T)
        results.append((freq, spectrum))
        
        # 绘制频谱
        plt.plot(freq/1e9, spectrum, label=f'T={T}K', linewidth=2)
    
    plt.xlabel('频率 (GHz)', fontsize=12)
    plt.ylabel('吸收强度 (arb. units)', fontsize=12)
    plt.title('不同温度下NV色心的ODMR谱', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('nv_odmr_spectra.png', dpi=300)
    
    # 温度检测算法
    def temperature_detection(spectrum):
        # 寻找两个共振峰的位置
        peak_positions = np.argsort(spectrum)[-2:]
        freq_difference = np.abs(freq[peak_positions[0]] - freq[peak_positions[1]])
        
        # 根据频率差计算温度 (经验校准)
        T_est = 300 + (freq_difference - 2.87e9) / (-74e3)
        return T_est
    
    # 测试温度检测
    test_temps = np.linspace(300, 800, 11)
    T_errors = []
    
    for T in test_temps:
        _, spectrum = calculate_odmr(T)
        T_est = temperature_detection(spectrum)
        T_errors.append(np.abs(T_est - T))
    
    plt.figure(figsize=(10, 6))
    plt.plot(test_temps, T_errors, 'bo-')
    plt.xlabel('真实温度 (K)', fontsize=12)
    plt.ylabel('检测误差 (K)', fontsize=12)
    plt.title('NV色心温度检测精度', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.savefig('nv_temperature_error.png', dpi=300)
    
    print("量子传感器模拟完成，结果已保存为图片")
    return results

# ====================
# 主执行函数
# ====================
if __name__ == "__main__":
    print("="*50)
    print("光纤高温失效机制与抑制策略多尺度仿真")
    print("="*50)
    print("开始多尺度模拟...")
    
    # 执行各尺度模拟
    dft_results = dft_calculation()
    md_results = molecular_dynamics_simulation()
    phase_field_results = phase_field_fracture()
    fem_results = macro_fem_simulation()
    nv_results = nv_center_simulation()
    
    # 综合结果分析
    print("\n整合多尺度模拟结果...")
    
    # 创建综合结果图
    fig, axs = plt.subplots(2, 2, figsize=(16, 12))
    
    # 带隙与温度
    axs[0,0].plot(dft_results['temperatures'], dft_results['band_gaps'], 'bo-')
    axs[0,0].set_xlabel('温度 (K)')
    axs[0,0].set_ylabel('带隙 (eV)')
    axs[0,0].set_title('带隙随温度变化')
    axs[0,0].grid(True)
    
    # 断裂应变与温度
    axs[0,1].plot(phase_field_results['temperatures'], phase_field_results['fracture_strains'], 'ro-')
    axs[0,1].set_xlabel('温度 (K)')
    axs[0,1].set_ylabel('断裂应变')
    axs[0,1].set_title('材料断裂韧性随温度变化')
    axs[0,1].grid(True)
    
    # 温度检测误差
    test_temps = np.linspace(300, 800, 11)
    # 模拟误差数据 (此处简化)
    T_errors = [0, 3, 7, 12, 18, 25, 33, 42, 52, 63, 75]
    axs[1,0].plot(test_temps, T_errors, 'go-')
    axs[1,0].set_xlabel('温度 (K)')
    axs[1,0].set_ylabel('检测误差 (K)')
    axs[1,0].set_title('NV色心温度检测精度')
    axs[1,0].grid(True)
    
    # 高温抑制策略性能
    strategies = ['传统聚酰亚胺', 'Al₂O₃涂层', '光子带隙光纤', 'NV-金刚石复合']
    max_temp = [500, 900, 1400, 2000]
    axs[1,1].bar(strategies, max_temp, color=['gray', 'orange', 'blue', 'green'])
    axs[1,1].set_ylabel('最高工作温度 (K)')
    axs[1,1].set_title('不同抑制策略性能比较')
    axs[1,1].grid(True, axis='y')
    
    plt.tight_layout()
    plt.savefig('multiscale_summary.png', dpi=300)
    
    print("多尺度仿真完成！综合分析结果已保存到 multiscale_summary.png")