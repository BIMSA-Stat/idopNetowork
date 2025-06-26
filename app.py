import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import eval_legendre as Legendre
from scipy.integrate import cumulative_trapezoid
from sklearn.model_selection import RandomizedSearchCV
from asgl import Regressor
from scipy.stats import uniform
import math
from igraph import Graph, plot

# 初始化session_state
if 'df' not in st.session_state:
    st.session_state.df = None
if 'time' not in st.session_state:
    st.session_state.time = None
if 'time_numbers' not in st.session_state:
    st.session_state.time_numbers = None
if 'variables_names' not in st.session_state:
    st.session_state.variables_names = None
if 'variables_numbers' not in st.session_state:
    st.session_state.variables_numbers = None
if 'Y_tilde' not in st.session_state:
    st.session_state.Y_tilde = None
if 'X' not in st.session_state:
    st.session_state.X = None
if 'X_integral' not in st.session_state:
    st.session_state.X_integral = None
if 'group_name_all' not in st.session_state:
    st.session_state.group_name_all = None
if 'group_inner_name_all' not in st.session_state:
    st.session_state.group_inner_name_all = None
if 'group_index' not in st.session_state:
    st.session_state.group_index = None
if 'custom_group_weights' not in st.session_state:
    st.session_state.custom_group_weights = None
if 'custom_individual_weights' not in st.session_state:
    st.session_state.custom_individual_weights = None
if 'group_selection_all' not in st.session_state:
    st.session_state.group_selection_all = None
if 'coef_asgl_list' not in st.session_state:
    st.session_state.coef_asgl_list = None
if 'coef_asgl_list_all' not in st.session_state:
    st.session_state.coef_asgl_list_all = None
if 'coef_asgl_group_all' not in st.session_state:
    st.session_state.coef_asgl_group_all = None
if 'best_params_list' not in st.session_state:
    st.session_state.best_params_list = None
if 'param_dist' not in st.session_state:
    st.session_state.param_dist = None
if 'X_columns' not in st.session_state:
    st.session_state.X_columns = None
if 'X_integral_columns' not in st.session_state:
    st.session_state.X_integral_columns = None
if 'Adjacency_matrix' not in st.session_state:
    st.session_state.Adjacency_matrix = None


# 应用名称设置 （标题）
st.title("idopNetwork_v0.1_yu")

data_file = st.file_uploader("上传数据文件", type=["csv"])
# param_dist = {'lambda1': [1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100, 1000, 10000],'alpha': uniform(loc=0, scale=1)} 
# param_dist = {'lambda1': [1e-3, 1e-2, 1e-1, 1, 10],'alpha': uniform(loc=0, scale=1)} 
# param_dist = {'lambda1': uniform(loc=0, scale=1),'alpha': uniform(loc=0, scale=1)} 
param_dist = {'lambda1': 10.0**np.arange(-3, 1.01, 0.2),'alpha': np.r_[np.arange(0.0, 0.3, 0.02), np.arange(0.3, 0.7, 0.1), np.arange(0.7, 0.99, 0.02)]} 


st.session_state.param_dist = param_dist

# 标签页设置
tab1, tab2, tab3, tab4 = st.tabs(["1. 数据加载", "2. 数据处理", "3. 变量选择","4. 绘制图形"])
with tab1:
    if st.button("查看数据"):
        if data_file is not None:
            # df = pd.read_csv(data_file, skiprows=range(1, 51), nrows=500, usecols=[0,2,3,4,5,6,7])
            df = pd.read_csv(data_file)  
            df.set_index('Time', inplace = True)            # 设置时间列作为index
            st.session_state.df = df
            st.success(f"数据加载成功！")
            time = df.index.values                          # 提取时间列
            time_numbers = time.shape[0]                    # 时间序列长度
            variables_names = df.columns.values             # 变量名称
            variables_numbers = variables_names.shape[0]    # 变量数量(列数)
            st.session_state.time = time                   # 将时间序列存储到session_state中
            st.session_state.time_numbers = time_numbers   # 将时间序列长度存储到session_state中
            st.session_state.variables_names = variables_names # 将变量名称存储到session_state中
            st.session_state.variables_numbers = variables_numbers # 将变量数量存储到session_state中
            st.write(f"时间序列长度: {time_numbers}")
            st.write(f"变量数量: {variables_numbers}")  # 变量数量(列数)  
            st.write(f"变量名称: {list(variables_names)}")  # 变量名称
            st.write(f"数据矩阵形状: {df.shape}")
            st.write(f"数据矩阵\n")
            st.dataframe(df)
        else:
            st.error("请先上传数据文件！")

with tab2:
    # 数据仿射变换
    if st.button("数据仿射变换"):
        if st.session_state.df is not None:
            Y = st.session_state.df.values
            # 将 Y 的每一列标准化(默认不执行)
            Y = (Y - Y.mean(axis=0)) / Y.std(axis=0)
            # 数据矩阵 Y 的第 i 个变量 (第 i 列) 仿射映射到 [-1,1] 区间
            def Y_tilde_fun(Y):
                Y_tilde = []
                for i in range(Y.shape[1]):
                    y_i = Y[:, i]
                    y_i_min = y_i.min()
                    y_i_max = y_i.max()
                    y_i_tilde = 2 * (y_i - y_i_min) / (y_i_max - y_i_min) - 1
                    Y_tilde.append(y_i_tilde)
                # 将列表转换为数组
                Y_tilde = np.array(Y_tilde).T
                return Y_tilde

            # 将数据矩阵 Y 的每个变量仿射映射到 [-1,1] 区间
            Y_tilde = Y_tilde_fun(Y) 
            st.session_state.Y_tilde = Y_tilde
            st.write(f"数据仿射变换成功！")
            st.write(f"数据仿射变换后的数据矩阵形状: {Y_tilde.shape}")
            st.write(f"数据仿射变换后的数据矩阵\n")
            st.dataframe(pd.DataFrame(Y_tilde ,columns=st.session_state.variables_names,index=st.session_state.time))
        else:
            st.warning("请先加载数据！")

    # 计算每一个变量的基展开,用户可以自定义阶数
    # 用户可以自定义阶数，默认为3
    n_order = st.slider("阶数", min_value=1, max_value=15, value=5, key="n_order")
    if st.button("计算每一个变量的基展开"):
        if st.session_state.Y_tilde is not None:
            # 第 i 个变量以勒让德多项式作为基底所得到的特征矩阵 (time_numbers 行，n_order+1 列)
            def X_i_fun(i, Y_tilde, n_order): 
                X_i = []
                for j in range(n_order+1):
                    # 计算j阶勒让德多项式在归一化变量上的值
                    X_i.append(Legendre(j, Y_tilde.T[i]))
                X_i = np.array(X_i).T
                return X_i

            # 所有变量以勒让德多项式作为基底所得到的特征矩阵 (time_numbers 行，variables_numbers*(n_order+1) 列的二维矩阵)
            X = np.zeros((st.session_state.time_numbers, st.session_state.variables_numbers*(n_order+1)))
            for i in range(st.session_state.variables_numbers):
                X[:, i*(n_order+1):(i+1)*(n_order+1)] = X_i_fun(i, st.session_state.Y_tilde, n_order)
            st.session_state.X = X
            
            # 为每个变量的每个阶数创建列名
            X_columns = []
            for var_name in st.session_state.variables_names:
                for j in range(n_order+1):
                    X_columns.append(f"{var_name}_order{j}")
            st.session_state.X_columns = X_columns
            st.write(f"计算每一个变量的基展开成功！")
            st.write(f"所有变量以勒让德多项式作为基底所得到的特征矩阵形状: {X.shape}")
            st.write(f"所有变量以勒让德多项式作为基底所得到的特征矩阵\n")
            st.dataframe(pd.DataFrame(X, columns=X_columns, index=st.session_state.time))
        else:
            st.warning("请先进行数据仿射变换！")

    # 计算每一个变量的勒让德多项式积分
    # 定义勒让德多项式的积分形式
    if st.button("计算每一个变量的勒让德多项式积分"):
        if st.session_state.Y_tilde is not None:
            def legendre_integral(y_tilde, n_order):
                integrals = []
                for k in range(n_order + 1):
                    # 第 k 阶勒让德多项式
                    legendre_k = Legendre(k, y_tilde)
                    # 对勒让德多项式进行积分
                    integral_k = cumulative_trapezoid(legendre_k, y_tilde, initial=0)
                    integrals.append(integral_k)
                return integrals
            
            # 对每个变量的每个勒让德多项式进行积分，得到特征矩阵 X_integral
            X_integral = np.zeros((st.session_state.time_numbers, st.session_state.variables_numbers*(n_order+1)))
            
            # 为积分结果创建列名
            X_integral_columns = []
            for var_name in st.session_state.variables_names:
                for j in range(n_order+1):
                    X_integral_columns.append(f"{var_name}_integral_order{j}")
            
            # 对每个变量的每个勒让德多项式进行积分
            for var_idx in range(st.session_state.variables_numbers):
                y_tilde = st.session_state.Y_tilde[:, var_idx]
                integrals = legendre_integral(y_tilde, n_order)
                # 将第 i 个变量的第 k 阶勒让德多项式积分结果 存储到 X_integral 的第 i*(n_order+1)+k 列
                for k in range(n_order + 1):
                    X_integral[:, var_idx * (n_order + 1) + k] = integrals[k]
            
            st.session_state.X_integral = X_integral
            st.write(f"计算每一个变量的勒让德多项式积分成功！")
            st.write(f"所有变量以勒让德多项式作为基底所得到的特征矩阵形状: {X_integral.shape}")
            st.write(f"所有变量以勒让德多项式作为基底所得到的特征矩阵\n")
            st.dataframe(pd.DataFrame(X_integral, columns=X_integral_columns, index=st.session_state.time))
        else:
            st.warning("请先进行数据仿射变换！")



with tab3:
    if st.session_state.df is not None:
        st.write(f"变量选择")
        def create_group_structures(variables_names, variables_numbers, n_order):
            def group_info_fun(i, variables_names, variables_numbers, n_order):
                group_name = []
                for j in range(variables_numbers):
                    if j == i:
                        group_name.append(f'{variables_names[i]}_self')
                    else:
                        group_name.append(f'{variables_names[j]} \u2192 {variables_names[i]}')
                group_inner_name = []
                for j in range(variables_numbers):
                    if j == i:
                        for k in range(n_order+1):
                            group_inner_name.append(f'{variables_names[i]}_self_order{k}')
                    else:
                        for k in range(n_order+1):
                            group_inner_name.append(f'{variables_names[j]} \u2192 {variables_names[i]}_order{k}')
                return group_name, group_inner_name
            group_name_all = []
            group_inner_name_all = []
            for i in range(variables_numbers):
                group_name, group_inner_name = group_info_fun(i, variables_names, variables_numbers, n_order)
                group_name_all.append(group_name)
                group_inner_name_all.append(group_inner_name)
            group_index = []
            for i in range(variables_numbers):
                group_index.extend([i] * (n_order+1)) 
            return group_name_all, group_inner_name_all, group_index

        group_name_all, group_inner_name_all, group_index = create_group_structures(st.session_state.variables_names, st.session_state.variables_numbers, n_order)
        st.session_state.group_name_all = group_name_all
        st.session_state.group_inner_name_all = group_inner_name_all
        st.session_state.group_index = group_index
        if st.button("查看组结构"):
            st.write(f"组名称: {group_name_all}")
            st.write(f"组内部名称: {group_inner_name_all}")
            st.write(f"组索引: {group_index}")      

        # 定义变量选择的权重(为了让自调控项的系数不为0，其它变量对当前变量的互作用项的系数为可以为0，表示被筛除)
        custom_group_weights = []
        for j in range(st.session_state.variables_numbers):
            row = [0.3] * st.session_state.variables_numbers # 创建全1的行
            row[j] = 0   # 在第j个位置设置为0
            custom_group_weights.append(row)

        custom_individual_weights = []
        for j in range(st.session_state.variables_numbers*(n_order+1)):
            row = [0.3] * st.session_state.variables_numbers*(n_order+1) # 创建全1的行
            row[j] = 0   # 在第j个位置设置为0
            custom_individual_weights.append(row)

        st.session_state.custom_group_weights = custom_group_weights
        st.session_state.custom_individual_weights = custom_individual_weights

        if st.button("执行变量选择"):
            if st.session_state.X_integral is not None:
                # 获取原始数据Y
                Y = st.session_state.df.values
                Y = (Y - Y.mean(axis=0)) / Y.std(axis=0)
                coef_asgl_list = []     # 单变量选择的系数结果
                coef_asgl_list_all = []   # 所有变量选择的系数结果
                coef_asgl_group_all = []  # 组选择的系数结果
                best_params_list = []    # 变量选择的最优参数

                for i in range(st.session_state.variables_numbers):
                    asgl_model_i = Regressor(model='lm',                    
                                penalization='asgl',                
                                individual_weights=custom_individual_weights[i],  
                                group_weights=custom_group_weights[i],       
                                individual_power_weight=1,            
                                group_power_weight=1,               
                                fit_intercept = False,               
                                tol=1e-3,                     
                                )
                    random_search = RandomizedSearchCV(
                        estimator=asgl_model_i,       
                        param_distributions=st.session_state.param_dist,   
                        n_iter=20,             
                        cv=5,                
                        verbose=1,             
                        n_jobs=-1              
                    )

                    random_search.fit(X=st.session_state.X_integral, y=Y[:,i], group_index=group_index)

                    best_params = random_search.best_params_
                    best_params_list.append(best_params)

                    best_model = random_search.best_estimator_
                    coef_asgl_list.append(best_model.coef_)
                    Y_pred = best_model.predict(st.session_state.X_integral)

                    coef_asgl_list_all.append(best_model.coef_)

                coef_asgl_list_all = np.array(coef_asgl_list_all).T

                group_selection_all = [] 
                for j in range(st.session_state.variables_numbers):
                    group_selection = []
                    for i in range(st.session_state.variables_numbers):
                        if np.any(coef_asgl_list[j].reshape(-1,1).T.reshape(st.session_state.variables_numbers,n_order+1)[i,:] != 0):
                            group_selection.append(1)
                        else:
                            group_selection.append(0)
                    group_selection_all.append(group_selection)
                group_selection_all = np.array(group_selection_all).T
                # 计算all_effects并存入session_state
                all_effects = []
                variables_numbers = st.session_state.variables_numbers
                time_numbers = st.session_state.time_numbers
                n_order = n_order  # 保证n_order取自slider
                X_integral = st.session_state.X_integral
                variables_names = st.session_state.variables_names
                time = st.session_state.time
                Y = st.session_state.df.values
                Y = (Y - Y.mean(axis=0)) / Y.std(axis=0)
                for eq_idx in range(variables_numbers):
                    effect_temp = []
                    for i in range(time_numbers):
                        effect_temp.append(np.array(coef_asgl_list_all.T[eq_idx])*np.array(X_integral[i,:]))
                    effect_temp = np.array(effect_temp)
                    Effect = np.zeros((time_numbers, variables_numbers))
                    for i in range(variables_numbers):
                        start_col = i * (n_order+1)
                        end_col = (i + 1) * (n_order+1)
                        Effect[:,i] = effect_temp[:,start_col:end_col].sum(axis=1)
                    Effect_total = np.sum(Effect,axis=1)
                    Effect = np.concatenate((Effect,Effect_total.reshape(-1,1)),axis=1)
                    all_effects.append(Effect)
                st.session_state.all_effects = all_effects
                st.session_state.group_selection_all = group_selection_all
                st.session_state.coef_asgl_list = coef_asgl_list
                st.session_state.coef_asgl_list_all = coef_asgl_list_all
                st.session_state.best_params_list = best_params_list
                
                # 网络的边权
                # 积分后的特征矩阵 X_integral 的列和
                X_integral_sum = np.array(st.session_state.X_integral.sum(axis=0)).reshape(-1,1).T

                # 每个变量的系数 乘以 X_integral 的列和
                effect = []
                for i in range(st.session_state.variables_numbers):
                    effect.append(np.array(coef_asgl_list_all.T[i])*np.array(X_integral_sum))
                effect = np.array(effect)
                # print("每个变量的系数 * 积分后的特征矩阵 X_integral 的列和")
                # print(tabulate(effect[0], headers=group_inner_name_all[0], tablefmt='p', showindex=False))
                # print(tabulate(effect[1], headers=group_inner_name_all[1], tablefmt='p', showindex=False))

                # 加权邻接矩阵
                Adjacency_matrix = []
                for i in range(st.session_state.variables_numbers):
                    Adjacency_matrix.append(np.sum(effect[i].reshape(st.session_state.variables_numbers,n_order+1), axis=1))
                Adjacency_matrix = np.array(Adjacency_matrix).T
                st.session_state.Adjacency_matrix = Adjacency_matrix
                
                st.success("变量选择完成！")
                st.write(f"最优参数: {best_params_list}")
                st.write(f"组选择结果: {group_selection_all}")
            else:
                st.warning("请先计算勒让德多项式积分！")
        
    else:
        st.warning("请先加载数据！")

with tab4:
    if st.session_state.df is not None:
        # 创建侧边栏
        with st.sidebar:
            st.header("图形配置")
            
            # 选择绘制模式
            plot_mode = st.selectbox(
                '选择绘制模式',
                ['单个变量', '全部变量'],
                key="plot_mode_selectbox"
            )
            
            # 图形类型选择
            plot_type = st.selectbox(
                '选择图形类型',
                ['网络图', '散点图', '直方图', '效应分解图'],
                key="plot_type_selectbox"
            )

            # 网络图专属参数
            if plot_type == '网络图':
                st.sidebar.markdown("### 网络图参数")
                edge_scale_input = st.sidebar.text_input('边的粗细缩放倍数', value='0.003')
                try:
                    edge_scale = float(edge_scale_input)
                except ValueError:
                    edge_scale = 0.003
                    st.sidebar.warning("请输入有效的数字，已使用默认值0.003")
                vertex_size = st.sidebar.slider('节点大小', min_value=10, max_value=100, value=50, step=1)
                vertex_label_size = st.sidebar.slider('节点标签字体', min_value=8, max_value=30, value=12, step=1)
                layout_mode = st.sidebar.selectbox('布局方式', ['circle', 'fr', 'kk'], index=0)
                edge_curved = st.sidebar.slider('边曲率', min_value=0.0, max_value=1.0, value=0.2, step=0.05)
            
            # 根据绘制模式显示不同的选项
            if plot_mode == '单个变量':
                column = st.selectbox('选择列', st.session_state.df.columns, key="sidebar_selectbox")
                
                # 如果是散点图，需要选择两个变量
                if plot_type == '散点图':
                    column2 = st.selectbox('选择Y轴列', st.session_state.df.columns, key="scatter_y_selectbox")
                
                # 如果是直方图，需要设置柱子数量
                if plot_type == '直方图':
                    bins = st.slider('柱子数量', min_value=5, max_value=50, value=10, key="histogram_bins_sidebar")
            else:  # 全部变量
                # 如果是散点图，需要选择两个变量
                if plot_type == '散点图':
                    column2 = st.selectbox('选择Y轴列', st.session_state.df.columns, key="scatter_y_selectbox")
                
                # 如果是直方图，需要设置柱子数量
                if plot_type == '直方图':
                    bins = st.slider('柱子数量', min_value=5, max_value=50, value=10, key="histogram_bins_sidebar")
            
            # 子图布局选择 - 默认选择网格布局
            subplot_layout = st.selectbox(
                '子图布局',
                ['网格布局', '单图显示', '垂直排列', '水平排列'],
                key="subplot_layout_selectbox"
            )
            
            # 如果是网格布局，选择行列数
            if subplot_layout == '网格布局':
                n_cols = st.slider('列数', 1, min(10, len(st.session_state.df.columns)), 2, key="grid_cols")
                n_rows = st.slider('行数', 1, min(10, len(st.session_state.df.columns)), 2, key="grid_rows")
            
            title = st.text_input('标题', f'{plot_type}', key="plot_title")
            x_label = st.text_input('X轴标签', '时间', key="plot_xlabel")
            y_label = st.text_input('Y轴标签', '数值', key="plot_ylabel")
            
            # 颜色选择 - 全部变量模式下可以选择颜色
            color = st.color_picker('颜色', '#1f77b4', key="plot_color")
            
            # 图形大小设置
            fig_width = st.slider('图形宽度', 6, 15, 10, key="fig_width")
            fig_height = st.slider('图形高度', 4, 10, 6, key="fig_height")
            
            # 网格线设置
            show_grid = st.checkbox('显示网格线', True, key="show_grid")
            
            # 图例设置
            show_legend = st.checkbox('显示图例', True, key="show_legend")
        
        # 主内容区域绘制图形
        if plot_type == '网络图':
            if st.session_state.Adjacency_matrix is not None:
                # 绘制Lasso网络图
                def plot_lasso_network(adj_matrix, vertex_names=None, edge_scale=0.003, vertex_size=50, vertex_label_size=12, layout_mode='circle', edge_curved=0.2):
                    """
                    绘制Lasso网络图，右侧显示窄条形出度入度对比图
                    
                    参数:
                    adj_matrix: 邻接矩阵
                    vertex_names: 节点名称列表，默认为None
                    """
                    if vertex_names is None:
                        vertex_names = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
                    
                    # 创建图
                    g = Graph.Weighted_Adjacency(adj_matrix, mode="directed")
                    g.vs["name"] = vertex_names
                    g.vs["label"] = vertex_names

                    # 删除自环边
                    g.simplify(loops=True)

                    # 获取布局
                    layout = g.layout(layout_mode)
                    # 旋转布局（仅对circle模式）
                    if layout_mode == 'circle':
                        angle = math.pi / 2
                        rotated_layout = [
                            (x * math.cos(angle) - y * math.sin(angle),
                            x * math.sin(angle) + y * math.cos(angle)) 
                            for x, y in layout
                        ]
                        rotated_layout = [rotated_layout[0]] + rotated_layout[:0:-1]
                    else:
                        rotated_layout = layout

                    # 根据边的权重设置颜色和宽度
                    if g.es.attributes() and "weight" in g.es.attributes():
                        edge_weights = g.es["weight"]
                    else:
                        edge_weights = []
                        for edge in g.es:
                            source = edge.source
                            target = edge.target
                            edge_weights.append(adj_matrix[source, target])
                    
                    edge_colors = []
                    edge_widths = []
                    
                    for weight in edge_weights:
                        if weight > 0:
                            edge_colors.append('red')
                        elif weight < 0:
                            edge_colors.append('blue')
                        else:
                            edge_colors.append('gray')
                        edge_widths.append(abs(weight) * edge_scale) 

                    visual_style = {
                        "vertex_size": vertex_size,
                        "vertex_color": "lightblue",
                        "vertex_label_size": vertex_label_size,
                        "edge_width": edge_widths,
                        "edge_color": edge_colors,
                        "layout": rotated_layout,
                        "bbox": (500, 500),
                        "margin": 50,
                        "edge_curved": edge_curved,
                    }

                    out_degrees = g.outdegree()
                    in_degrees = g.indegree()
                    max_degree = max(max(in_degrees), max(out_degrees))
                    y_pos = np.arange(len(vertex_names))

                    fig = plt.figure(figsize=(11, 8))
                    gs = fig.add_gridspec(1, 2, width_ratios=[3, 1])

                    ax1 = fig.add_subplot(gs[0])
                    plot(g, target=ax1, **visual_style)
                    ax1.set_title('Lasso Network', fontsize=16)
                    ax1.axis('off')

                    ax2 = fig.add_subplot(gs[1])
                    bar_width = 0.6
                    ax2.barh(y_pos, [-d for d in in_degrees], height=bar_width, 
                        color='lightgreen', label='In-degree')
                    ax2.barh(y_pos, out_degrees, height=bar_width, 
                        color='skyblue', label='Out-degree', left=0)
                    ax2.axvline(x=0, color='gray', linestyle='--', linewidth=0.8)
                    ax2.set_yticks(y_pos)
                    ax2.set_yticklabels(vertex_names, fontsize=12)
                    ax2.set_xticks([])
                    ax2.spines['top'].set_visible(False)
                    ax2.spines['right'].set_visible(False)
                    ax2.spines['bottom'].set_visible(False)
                    ax2.spines['left'].set_visible(False)
                    ax2.legend(loc='upper right', fontsize=8, framealpha=0.5)
                    plt.tight_layout()
                    plt.subplots_adjust(wspace=0.05)
                    return fig

                # 绘制网络图
                fig = plot_lasso_network(
                    st.session_state.Adjacency_matrix,
                    st.session_state.variables_names,
                    edge_scale=edge_scale,
                    vertex_size=vertex_size,
                    vertex_label_size=vertex_label_size,
                    layout_mode=layout_mode,
                    edge_curved=edge_curved
                )
                st.pyplot(fig)
                
                # 显示邻接矩阵
                st.write("邻接矩阵:")
                st.dataframe(pd.DataFrame(st.session_state.Adjacency_matrix, 
                                        columns=st.session_state.variables_names,
                                        index=st.session_state.variables_names))
            else:
                st.warning("请先执行变量选择以生成邻接矩阵！")
        
        elif plot_type == '效应分解图':
            all_effects = st.session_state.all_effects
            variables_numbers = st.session_state.variables_numbers
            variables_names = st.session_state.variables_names
            time = st.session_state.time
            Y = st.session_state.df.values
            Y = (Y - Y.mean(axis=0)) / Y.std(axis=0)
            if all_effects is not None:
                fig, axes = plt.subplots(variables_numbers, 1, figsize=(10, 4*variables_numbers))
                if variables_numbers == 1:
                    axes = [axes]
                for i in range(variables_numbers):
                    ax = axes[i]
                    independent_effect = all_effects[i][:,i]
                    if np.any(independent_effect != 0):
                        ax.plot(time, independent_effect, color='red', linewidth=1, label=f'{variables_names[i]} independent')
                    ax.scatter(time, Y[:,i], color='black', alpha=0.3, s=8, label='true')
                    for j in range(variables_numbers):
                        if j != i:
                            dependent_effect = all_effects[i][:,j]
                            if np.any(dependent_effect != 0):
                                ax.plot(time, dependent_effect, color='green', alpha=0.6, linewidth=2, label=f'{variables_names[j]} dependent')
                    total_effect = all_effects[i][:,variables_numbers]
                    if np.any(total_effect != 0):
                        ax.plot(time, total_effect, color='blue', linewidth=1, label='total')
                    ax.legend(loc='upper right', fontsize=8, framealpha=0.8)
                    ax.set_title(f'Effect Decomposition for {variables_names[i]}', fontsize=10)
                    ax.set_xlabel('Time', fontsize=9)
                    ax.set_ylabel('Effect', fontsize=9)
                plt.tight_layout()
                st.pyplot(fig)
            else:
                st.warning('请先执行变量选择以生成效应分解结果！')
        
        elif plot_mode == '单个变量' or (plot_mode == '全部变量' and subplot_layout == '单图显示'):
            fig, ax = plt.subplots(figsize=(fig_width, fig_height))
            
            if plot_mode == '单个变量':
                if plot_type == '散点图':
                    ax.scatter(st.session_state.time, st.session_state.df[column], color=color, alpha=0.6, s=30, label=f'{column} vs {column2}')
                    ax.set_title(title)
                    ax.set_xlabel(x_label)
                    ax.set_ylabel(y_label)
                
                elif plot_type == '直方图':
                    ax.hist(st.session_state.df[column], bins=bins, color=color, alpha=0.7, edgecolor='black', label=column)
                    ax.set_title(title)
                    ax.set_xlabel(x_label)
                    ax.set_ylabel(y_label)
                    
                    # 添加均值线
                    mean_val = st.session_state.df[column].mean()
                    ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'均值: {mean_val:.2f}')
            
            else:  # 全部变量 - 单图显示
                if plot_type == '散点图':
                    # 绘制所有变量的时间序列
                    for i, col in enumerate(st.session_state.df.columns):
                        ax.scatter(st.session_state.time, st.session_state.df[col], 
                                 color=color, alpha=0.6, s=30, label=col)
                    ax.set_title(title)
                    ax.set_xlabel(x_label)
                    ax.set_ylabel(y_label)
                
                elif plot_type == '直方图':
                    # 绘制所有变量的直方图
                    for i, col in enumerate(st.session_state.df.columns):
                        ax.hist(st.session_state.df[col], bins=bins, color=color, alpha=0.5, 
                               edgecolor='black', label=col)
                    ax.set_title(title)
                    ax.set_xlabel(x_label)
                    ax.set_ylabel(y_label)
            
            # 通用设置
            if show_grid:
                ax.grid(True, alpha=0.3)
            
            if show_legend:
                ax.legend()
            
            # Rotate X-axis labels for time series
            if plot_type not in ['箱线图', '直方图']:
                plt.xticks(rotation=45)
            
            plt.tight_layout()
            st.pyplot(fig)
        
        else:  # 全部变量 - 子图布局
            if plot_type == '散点图':
                # 子图布局
                if subplot_layout == '网格布局':
                    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_width*2, fig_height*2))
                    axes = axes.flatten() if n_rows > 1 or n_cols > 1 else [axes]
                elif subplot_layout == '垂直排列':
                    fig, axes = plt.subplots(len(st.session_state.df.columns), 1, figsize=(fig_width, fig_height*len(st.session_state.df.columns)))
                    axes = axes.flatten() if len(st.session_state.df.columns) > 1 else [axes]
                else:  # 水平排列
                    fig, axes = plt.subplots(1, len(st.session_state.df.columns), figsize=(fig_width*len(st.session_state.df.columns), fig_height))
                    axes = axes.flatten() if len(st.session_state.df.columns) > 1 else [axes]
                
                for i, col in enumerate(st.session_state.df.columns):
                    if i < len(axes):
                        axes[i].scatter(st.session_state.time, st.session_state.df[col], 
                                      color=color, alpha=0.6, s=30)
                        axes[i].set_title(f'{col} 时间序列')
                        axes[i].set_xlabel('时间')
                        axes[i].set_ylabel(col)
                        if show_grid:
                            axes[i].grid(True, alpha=0.3)
                
                # 隐藏多余的子图
                for i in range(len(st.session_state.df.columns), len(axes)):
                    axes[i].set_visible(False)
                
                plt.tight_layout()
                st.pyplot(fig)
            
            elif plot_type == '直方图':
                # 子图布局
                if subplot_layout == '网格布局':
                    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_width*2, fig_height*2))
                    axes = axes.flatten() if n_rows > 1 or n_cols > 1 else [axes]
                elif subplot_layout == '垂直排列':
                    fig, axes = plt.subplots(len(st.session_state.df.columns), 1, figsize=(fig_width, fig_height*len(st.session_state.df.columns)))
                    axes = axes.flatten() if len(st.session_state.df.columns) > 1 else [axes]
                else:  # 水平排列
                    fig, axes = plt.subplots(1, len(st.session_state.df.columns), figsize=(fig_width*len(st.session_state.df.columns), fig_height))
                    axes = axes.flatten() if len(st.session_state.df.columns) > 1 else [axes]
                
                for i, col in enumerate(st.session_state.df.columns):
                    if i < len(axes):
                        axes[i].hist(st.session_state.df[col], bins=bins, color=color, alpha=0.7, 
                                   edgecolor='black')
                        axes[i].set_title(f'{col} 直方图')
                        axes[i].set_xlabel(col)
                        axes[i].set_ylabel('频数')
                        
                        # 添加均值线
                        mean_val = st.session_state.df[col].mean()
                        axes[i].axvline(mean_val, color='red', linestyle='--', linewidth=2, 
                                      label=f'均值: {mean_val:.2f}')
                        
                        if show_grid:
                            axes[i].grid(True, alpha=0.3)
                        if show_legend:
                            axes[i].legend()
                
                # 隐藏多余的子图
                for i in range(len(st.session_state.df.columns), len(axes)):
                    axes[i].set_visible(False)
                
                plt.tight_layout()
                st.pyplot(fig)
        
    else:
        st.warning("请先加载数据！")
