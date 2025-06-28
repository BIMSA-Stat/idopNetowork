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
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# 页面配置
st.set_page_config(
    page_title="idopNetwork 数据分析平台",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自定义CSS样式
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .step-header {
        font-size: 1.5rem;
        color: #2e8b57;
        margin: 1rem 0;
        border-left: 4px solid #2e8b57;
        padding-left: 1rem;
    }
    .status-box {
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .status-success {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
    }
    .status-warning {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        color: #856404;
    }
    .status-info {
        background-color: #d1ecf1;
        border: 1px solid #bee5eb;
        color: #0c5460;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 0.5rem;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    .progress-step {
        display: flex;
        align-items: center;
        margin: 0.5rem 0;
        padding: 0.5rem;
        border-radius: 0.25rem;
    }
    .progress-step.completed {
        background-color: #d4edda;
        color: #155724;
    }
    .progress-step.current {
        background-color: #fff3cd;
        color: #856404;
    }
    .progress-step.pending {
        background-color: #f8f9fa;
        color: #6c757d;
    }
</style>
""", unsafe_allow_html=True)

# 初始化session_state
def init_session_state():
    """初始化所有session state变量"""
    defaults = {
        'df': None, 'time': None, 'time_numbers': None, 'variables_names': None,
        'variables_numbers': None, 'Y_tilde': None, 'X': None, 'X_integral': None,
        'group_name_all': None, 'group_inner_name_all': None, 'group_index': None,
        'custom_group_weights': None, 'custom_individual_weights': None,
        'group_selection_all': None, 'coef_asgl_list': None, 'coef_asgl_list_all': None,
        'coef_asgl_group_all': None, 'best_params_list': None, 'param_dist': None,
        'X_columns': None, 'X_integral_columns': None, 'Adjacency_matrix': None,
        'all_effects': None, 'analysis_complete': False, 'current_step': 1
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

# 进度跟踪
def get_progress_status():
    """获取当前分析进度状态"""
    steps = [
        {"name": "数据加载", "key": "df", "icon": "📁"},
        {"name": "数据预处理", "key": "Y_tilde", "icon": "🔄"},
        {"name": "特征工程", "key": "X_integral", "icon": "⚙️"},
        {"name": "变量选择", "key": "Adjacency_matrix", "icon": "🎯"},
        {"name": "结果可视化", "key": "analysis_complete", "icon": "📊"}
    ]
    
    completed_steps = 0
    for step in steps:
        if st.session_state.get(step["key"]) is not None and st.session_state.get(step["key"]) is not False:
            completed_steps += 1
        else:
            break
    
    return steps, completed_steps

def display_progress():
    """显示进度条"""
    steps, completed = get_progress_status()
    progress = completed / len(steps)
    
    st.sidebar.markdown("### 📈 分析进度")
    st.sidebar.progress(progress)
    st.sidebar.markdown(f"**{completed}/{len(steps)} 步骤完成**")
    
    for i, step in enumerate(steps):
        if i < completed:
            status_class = "completed"
        elif i == completed:
            status_class = "current"
        else:
            status_class = "pending"
        
        st.sidebar.markdown(f"""
        <div class="progress-step {status_class}">
            {step['icon']} {step['name']}
        </div>
        """, unsafe_allow_html=True)

# 数据统计卡片
def display_data_metrics():
    """显示数据统计信息"""
    if st.session_state.df is not None:
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <h3>📊 样本数量</h3>
                <h2>{st.session_state.time_numbers}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <h3>🔢 变量数量</h3>
                <h2>{st.session_state.variables_numbers}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            missing_values = st.session_state.df.isnull().sum().sum()
            st.markdown(f"""
            <div class="metric-card">
                <h3>❓ 缺失值</h3>
                <h2>{missing_values}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            data_range = f"{st.session_state.df.values.min():.2f} ~ {st.session_state.df.values.max():.2f}"
            st.markdown(f"""
            <div class="metric-card">
                <h3>📏 数据范围</h3>
                <h2>{data_range}</h2>
            </div>
            """, unsafe_allow_html=True)

# 初始化
init_session_state()

# 主标题
st.markdown('<h1 class="main-header">🧬 idopNetwork 数据分析平台 v0.2</h1>', unsafe_allow_html=True)

# 侧边栏配置
with st.sidebar:
    st.markdown("### ⚙️ 分析参数配置")
    
    # 全局参数设置
    with st.expander("🔧 全局参数", expanded=True):
        n_order = st.slider("勒让德多项式阶数", min_value=1, max_value=15, value=5, help="控制特征复杂度")
        cv_folds = st.slider("交叉验证折数", min_value=3, max_value=10, value=5)
        n_iter = st.slider("随机搜索迭代次数", min_value=10, max_value=100, value=20)
    
    # 可视化参数
    with st.expander("🎨 可视化参数"):
        color_theme = st.selectbox("颜色主题", ["默认", "深色", "彩虹", "专业"], index=0)
        fig_size = st.selectbox("图形大小", ["小", "中", "大"], index=1)
        show_confidence = st.checkbox("显示置信区间", value=True)
    
    # 显示进度
    display_progress()
    
    # 数据导出选项
    if st.session_state.Adjacency_matrix is not None:
        st.markdown("### 💾 导出选项")
        if st.button("导出邻接矩阵", key="sidebar_export_adj"):
            csv = pd.DataFrame(st.session_state.Adjacency_matrix, 
                             columns=st.session_state.variables_names,
                             index=st.session_state.variables_names).to_csv()
            st.download_button("下载CSV", csv, "adjacency_matrix.csv", "text/csv", key="sidebar_download_adj")

# 主要参数配置
param_dist = {
    'lambda1': 10.0**np.arange(-3, 1.01, 0.2),
    'alpha': np.r_[np.arange(0.0, 0.3, 0.02), np.arange(0.3, 0.7, 0.1), np.arange(0.7, 0.99, 0.02)]
}
st.session_state.param_dist = param_dist

# 主内容区域
tab1, tab2, tab3, tab4, tab5 = st.tabs(["📁 数据加载", "🔄 数据预处理", "⚙️ 特征工程", "🎯 变量选择", "📊 结果分析"])

with tab1:
    st.markdown('<div class="step-header">📁 步骤1: 数据加载与预览</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("#### 📋 上传数据文件")
        data_file = st.file_uploader(
            "支持CSV格式文件", 
            type=["csv"],
            help="请确保第一列为时间列，命名为'Time'"
        )
        
        if st.button("🔍 加载并分析数据", type="primary", key="load_data_btn"):
            if data_file is not None:
                try:
                    with st.spinner("正在加载数据..."):
                        df = pd.read_csv(data_file)
                        df.set_index('Time', inplace=True)
                        st.session_state.df = df
                        
                        # 提取基本信息
                        st.session_state.time = df.index.values
                        st.session_state.time_numbers = df.shape[0]
                        st.session_state.variables_names = df.columns.values
                        st.session_state.variables_numbers = df.shape[1]
                    
                    st.markdown('<div class="status-box status-success">✅ 数据加载成功！</div>', unsafe_allow_html=True)
                    
                except Exception as e:
                    st.markdown(f'<div class="status-box status-warning">❌ 数据加载失败: {str(e)}</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="status-box status-warning">⚠️ 请先选择数据文件</div>', unsafe_allow_html=True)
    
    with col2:
        if st.session_state.df is not None:
            st.markdown("#### 📈 数据质量检查")
            quality_checks = []
            
            # 检查缺失值
            missing_ratio = st.session_state.df.isnull().sum().sum() / (st.session_state.df.shape[0] * st.session_state.df.shape[1])
            if missing_ratio == 0:
                quality_checks.append("✅ 无缺失值")
            else:
                quality_checks.append(f"⚠️ 缺失值比例: {missing_ratio:.2%}")
            
            # 检查数据类型
            numeric_cols = st.session_state.df.select_dtypes(include=[np.number]).shape[1]
            if numeric_cols == st.session_state.df.shape[1]:
                quality_checks.append("✅ 全部为数值型数据")
            else:
                quality_checks.append(f"⚠️ 非数值列: {st.session_state.df.shape[1] - numeric_cols}")
            
            # 检查异常值
            has_outliers = False
            for col in st.session_state.df.columns:
                Q1 = st.session_state.df[col].quantile(0.25)
                Q3 = st.session_state.df[col].quantile(0.75)
                IQR = Q3 - Q1
                outliers = ((st.session_state.df[col] < (Q1 - 1.5 * IQR)) | 
                           (st.session_state.df[col] > (Q3 + 1.5 * IQR))).sum()
                if outliers > 0:
                    has_outliers = True
                    break
            
            if not has_outliers:
                quality_checks.append("✅ 无明显异常值")
            else:
                quality_checks.append("⚠️ 检测到异常值")
            
            for check in quality_checks:
                st.write(check)
    
    # 数据预览
    if st.session_state.df is not None:
        display_data_metrics()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 📋 数据概览")
            st.dataframe(st.session_state.df.head(10), use_container_width=True)
        
        with col2:
            st.markdown("#### 📊 统计摘要")
            st.dataframe(st.session_state.df.describe(), use_container_width=True)
        
        # 数据可视化预览
        st.markdown("#### 📈 数据可视化预览")
        
        # 创建交互式时间序列图
        df_reset = st.session_state.df.reset_index()
        fig = px.line(df_reset, x='Time', 
                      y=st.session_state.df.columns.tolist(),
                      title="所有变量的时间序列")
        fig.update_layout(height=400, xaxis_title="时间")
        st.plotly_chart(fig, use_container_width=True)
        
        # 相关性热力图
        corr_matrix = st.session_state.df.corr()
        fig_corr = px.imshow(corr_matrix, 
                            x=corr_matrix.columns, 
                            y=corr_matrix.columns,
                            title="变量相关性热力图",
                            color_continuous_scale="RdBu_r")
        fig_corr.update_layout(height=400)
        st.plotly_chart(fig_corr, use_container_width=True)

with tab2:
    st.markdown('<div class="step-header">🔄 步骤2: 数据预处理</div>', unsafe_allow_html=True)
    
    if st.session_state.df is None:
        st.markdown('<div class="status-box status-warning">⚠️ 请先在"数据加载"标签页中加载数据</div>', unsafe_allow_html=True)
    else:
        col1, col2 = st.columns([3, 1])
        
        with col2:
            st.markdown("#### 🔧 预处理选项")
            normalize_data = st.checkbox("标准化数据", value=True, help="将数据标准化为均值0，标准差1")
            affine_transform = st.checkbox("仿射变换", value=True, help="将数据映射到[-1,1]区间")
            
        with col1:
            if st.button("🚀 开始数据预处理", type="primary", key="preprocess_data_btn"):
                with st.spinner("正在进行数据预处理..."):
                    Y = st.session_state.df.values
                    
                    # 标准化
                    if normalize_data:
                        Y = (Y - Y.mean(axis=0)) / Y.std(axis=0)
                        st.success("✅ 数据标准化完成")
                    
                    # 仿射变换
                    if affine_transform:
                        def Y_tilde_fun(Y):
                            Y_tilde = []
                            for i in range(Y.shape[1]):
                                y_i = Y[:, i]
                                y_i_min = y_i.min()
                                y_i_max = y_i.max()
                                y_i_tilde = 2 * (y_i - y_i_min) / (y_i_max - y_i_min) - 1
                                Y_tilde.append(y_i_tilde)
                            return np.array(Y_tilde).T
                        
                        Y_tilde = Y_tilde_fun(Y)
                        st.session_state.Y_tilde = Y_tilde
                        st.success("✅ 仿射变换完成")
                        
                        st.markdown('<div class="status-box status-success">✅ 数据预处理完成！</div>', unsafe_allow_html=True)
        
        # 显示预处理结果
        if st.session_state.Y_tilde is not None:
            st.markdown("#### 📊 预处理结果对比")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("##### 原始数据分布")
                original_df = pd.DataFrame(st.session_state.df.values, 
                                         columns=st.session_state.variables_names)
                fig_orig = px.box(original_df, title="原始数据箱线图")
                st.plotly_chart(fig_orig, use_container_width=True)
            
            with col2:
                st.markdown("##### 预处理后数据分布")
                processed_df = pd.DataFrame(st.session_state.Y_tilde, 
                                          columns=st.session_state.variables_names)
                fig_proc = px.box(processed_df, title="预处理后数据箱线图")
                st.plotly_chart(fig_proc, use_container_width=True)
            
            # 显示预处理后的数据表格
            st.markdown("#### 📋 预处理后的数据")
            processed_display_df = pd.DataFrame(st.session_state.Y_tilde,
                                              columns=st.session_state.variables_names,
                                              index=st.session_state.time)
            st.dataframe(processed_display_df.head(10), use_container_width=True)

with tab3:
    st.markdown('<div class="step-header">⚙️ 步骤3: 特征工程</div>', unsafe_allow_html=True)
    
    if st.session_state.Y_tilde is None:
        st.markdown('<div class="status-box status-warning">⚠️ 请先完成数据预处理</div>', unsafe_allow_html=True)
    else:
        col1, col2 = st.columns([3, 1])
        
        with col2:
            st.markdown("#### 🔧 特征工程参数")
            st.info(f"当前阶数: {n_order}")
            total_features = st.session_state.variables_numbers * (n_order + 1)
            st.metric("预计特征数量", total_features)
        
        with col1:
            if st.button("🔬 开始特征工程", type="primary", key="feature_engineering_btn"):
                with st.spinner("正在计算勒让德多项式基展开..."):
                    # 计算基展开
                    def X_i_fun(i, Y_tilde, n_order): 
                        X_i = []
                        for j in range(n_order+1):
                            X_i.append(Legendre(j, Y_tilde.T[i]))
                        return np.array(X_i).T

                    X = np.zeros((st.session_state.time_numbers, st.session_state.variables_numbers*(n_order+1)))
                    for i in range(st.session_state.variables_numbers):
                        X[:, i*(n_order+1):(i+1)*(n_order+1)] = X_i_fun(i, st.session_state.Y_tilde, n_order)
                    st.session_state.X = X
                    
                    # 创建列名
                    X_columns = []
                    for var_name in st.session_state.variables_names:
                        for j in range(n_order+1):
                            X_columns.append(f"{var_name}_order{j}")
                    st.session_state.X_columns = X_columns
                    
                    st.success("✅ 基展开计算完成")
                
                with st.spinner("正在计算勒让德多项式积分..."):
                    def legendre_integral(y_tilde, n_order):
                        integrals = []
                        for k in range(n_order + 1):
                            legendre_k = Legendre(k, y_tilde)
                            integral_k = cumulative_trapezoid(legendre_k, y_tilde, initial=0)
                            integrals.append(integral_k)
                        return integrals
                    
                    X_integral = np.zeros((st.session_state.time_numbers, st.session_state.variables_numbers*(n_order+1)))
                    X_integral_columns = []
                    
                    for var_name in st.session_state.variables_names:
                        for j in range(n_order+1):
                            X_integral_columns.append(f"{var_name}_integral_order{j}")
                    
                    for var_idx in range(st.session_state.variables_numbers):
                        y_tilde = st.session_state.Y_tilde[:, var_idx]
                        integrals = legendre_integral(y_tilde, n_order)
                        for k in range(n_order + 1):
                            X_integral[:, var_idx * (n_order + 1) + k] = integrals[k]
                    
                    st.session_state.X_integral = X_integral
                    st.session_state.X_integral_columns = X_integral_columns
                    
                    st.success("✅ 积分特征计算完成")
                    
                    # 计算组结构
                    def create_group_structures(variables_names, variables_numbers, n_order):
                        def group_info_fun(i, variables_names, variables_numbers, n_order):
                            group_name = []
                            for j in range(variables_numbers):
                                if j == i:
                                    group_name.append(f'{variables_names[i]}_self')
                                else:
                                    group_name.append(f'{variables_names[j]} → {variables_names[i]}')
                            group_inner_name = []
                            for j in range(variables_numbers):
                                if j == i:
                                    for k in range(n_order+1):
                                        group_inner_name.append(f'{variables_names[i]}_self_order{k}')
                                else:
                                    for k in range(n_order+1):
                                        group_inner_name.append(f'{variables_names[j]} → {variables_names[i]}_order{k}')
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

                    group_name_all, group_inner_name_all, group_index = create_group_structures(
                        st.session_state.variables_names, st.session_state.variables_numbers, n_order)
                    
                    st.session_state.group_name_all = group_name_all
                    st.session_state.group_inner_name_all = group_inner_name_all
                    st.session_state.group_index = group_index
                    
                    st.success("✅ 组结构计算完成")
                    st.markdown('<div class="status-box status-success">✅ 特征工程完成！</div>', unsafe_allow_html=True)
        
        # 显示特征工程结果
        if st.session_state.X_integral is not None:
            st.markdown("#### 📊 特征工程结果")
            
            tab3_1, tab3_2, tab3_3 = st.tabs(["基展开结果", "积分特征", "特征统计"])
            
            with tab3_1:
                st.markdown("##### 勒让德多项式基展开矩阵")
                X_df = pd.DataFrame(st.session_state.X, 
                                  columns=st.session_state.X_columns, 
                                  index=st.session_state.time)
                st.dataframe(X_df.head(10), use_container_width=True)
                
                # 可视化基展开
                X_df_reset = X_df.reset_index()
                x_col = X_df_reset.columns[0]  # 获取第一列作为x轴（时间轴）
                fig_basis = px.line(X_df_reset.iloc[:100], x=x_col, 
                                   y=st.session_state.X_columns[:min(6, len(st.session_state.X_columns))],
                                   title="前6个基函数展开结果")
                fig_basis.update_layout(xaxis_title="时间")
                st.plotly_chart(fig_basis, use_container_width=True)
            
            with tab3_2:
                st.markdown("##### 勒让德多项式积分特征矩阵")
                X_integral_df = pd.DataFrame(st.session_state.X_integral, 
                                           columns=st.session_state.X_integral_columns, 
                                           index=st.session_state.time)
                st.dataframe(X_integral_df.head(10), use_container_width=True)
                
                # 可视化积分特征
                X_integral_df_reset = X_integral_df.reset_index()
                x_col = X_integral_df_reset.columns[0]  # 获取第一列作为x轴（时间轴）
                fig_integral = px.line(X_integral_df_reset.iloc[:100], x=x_col, 
                                     y=st.session_state.X_integral_columns[:min(6, len(st.session_state.X_integral_columns))],
                                     title="前6个积分特征结果")
                fig_integral.update_layout(xaxis_title="时间")
                st.plotly_chart(fig_integral, use_container_width=True)
            
            with tab3_3:
                st.markdown("##### 特征统计信息")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("基展开特征数量", st.session_state.X.shape[1])
                    st.metric("积分特征数量", st.session_state.X_integral.shape[1])
                
                with col2:
                    st.metric("样本数量", st.session_state.X.shape[0])
                    feature_density = np.count_nonzero(st.session_state.X_integral) / st.session_state.X_integral.size
                    st.metric("特征密度", f"{feature_density:.3f}")

with tab4:
    st.markdown('<div class="step-header">🎯 步骤4: 变量选择</div>', unsafe_allow_html=True)
    
    if st.session_state.X_integral is None:
        st.markdown('<div class="status-box status-warning">⚠️ 请先完成特征工程</div>', unsafe_allow_html=True)
    else:
        # 准备权重矩阵
        if st.session_state.custom_group_weights is None:
            custom_group_weights = []
            for j in range(st.session_state.variables_numbers):
                row = [0.3] * st.session_state.variables_numbers
                row[j] = 0   # 自身权重设为0
                custom_group_weights.append(row)
            st.session_state.custom_group_weights = custom_group_weights

        if st.session_state.custom_individual_weights is None:
            custom_individual_weights = []
            for j in range(st.session_state.variables_numbers*(n_order+1)):
                row = [0.3] * st.session_state.variables_numbers*(n_order+1)
                row[j] = 0
                custom_individual_weights.append(row)
            st.session_state.custom_individual_weights = custom_individual_weights
        
        col1, col2 = st.columns([3, 1])
        
        with col2:
            st.markdown("#### 🔧 建模参数")
            st.info(f"交叉验证折数: {cv_folds}")
            st.info(f"随机搜索次数: {n_iter}")
            
            total_models = st.session_state.variables_numbers
            st.metric("模型数量", total_models)
            
            estimated_time = total_models * n_iter * cv_folds * 0.1
            st.metric("预估时间", f"{estimated_time:.1f}秒")
        
        with col1:
            if st.button("🚀 开始变量选择", type="primary", key="variable_selection_btn"):
                with st.spinner("正在执行变量选择算法...这可能需要几分钟时间"):
                    Y = st.session_state.df.values
                    Y = (Y - Y.mean(axis=0)) / Y.std(axis=0)
                    
                    coef_asgl_list = []
                    coef_asgl_list_all = []
                    best_params_list = []
                    
                    # 创建进度条
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    for i in range(st.session_state.variables_numbers):
                        status_text.text(f'正在处理第 {i+1}/{st.session_state.variables_numbers} 个变量: {st.session_state.variables_names[i]}')
                        
                        asgl_model_i = Regressor(
                            model='lm',                    
                            penalization='asgl',                
                            individual_weights=st.session_state.custom_individual_weights[i],  
                            group_weights=st.session_state.custom_group_weights[i],       
                            individual_power_weight=1,            
                            group_power_weight=1,               
                            fit_intercept=False,               
                            tol=1e-3,                     
                        )
                        
                        random_search = RandomizedSearchCV(
                            estimator=asgl_model_i,       
                            param_distributions=st.session_state.param_dist,   
                            n_iter=n_iter,             
                            cv=cv_folds,                
                            verbose=0,             
                            n_jobs=-1              
                        )

                        random_search.fit(X=st.session_state.X_integral, y=Y[:,i], group_index=st.session_state.group_index)

                        best_params = random_search.best_params_
                        best_params_list.append(best_params)

                        best_model = random_search.best_estimator_
                        coef_asgl_list.append(best_model.coef_)
                        coef_asgl_list_all.append(best_model.coef_)
                        
                        # 更新进度
                        progress_bar.progress((i + 1) / st.session_state.variables_numbers)

                    status_text.text('正在计算最终结果...')
                    
                    coef_asgl_list_all = np.array(coef_asgl_list_all).T

                    # 计算组选择结果
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
                    
                    # 计算效应分解
                    all_effects = []
                    for eq_idx in range(st.session_state.variables_numbers):
                        effect_temp = []
                        for i in range(st.session_state.time_numbers):
                            effect_temp.append(np.array(coef_asgl_list_all.T[eq_idx])*np.array(st.session_state.X_integral[i,:]))
                        effect_temp = np.array(effect_temp)
                        Effect = np.zeros((st.session_state.time_numbers, st.session_state.variables_numbers))
                        for i in range(st.session_state.variables_numbers):
                            start_col = i * (n_order+1)
                            end_col = (i + 1) * (n_order+1)
                            Effect[:,i] = effect_temp[:,start_col:end_col].sum(axis=1)
                        Effect_total = np.sum(Effect,axis=1)
                        Effect = np.concatenate((Effect,Effect_total.reshape(-1,1)),axis=1)
                        all_effects.append(Effect)
                    
                    # 计算邻接矩阵
                    X_integral_sum = np.array(st.session_state.X_integral.sum(axis=0)).reshape(-1,1).T
                    effect = []
                    for i in range(st.session_state.variables_numbers):
                        effect.append(np.array(coef_asgl_list_all.T[i])*np.array(X_integral_sum))
                    effect = np.array(effect)

                    Adjacency_matrix = []
                    for i in range(st.session_state.variables_numbers):
                        Adjacency_matrix.append(np.sum(effect[i].reshape(st.session_state.variables_numbers,n_order+1), axis=1))
                    Adjacency_matrix = np.array(Adjacency_matrix).T
                    
                    # 保存结果
                    st.session_state.all_effects = all_effects
                    st.session_state.group_selection_all = group_selection_all
                    st.session_state.coef_asgl_list = coef_asgl_list
                    st.session_state.coef_asgl_list_all = coef_asgl_list_all
                    st.session_state.best_params_list = best_params_list
                    st.session_state.Adjacency_matrix = Adjacency_matrix
                    
                    progress_bar.progress(1.0)
                    status_text.text('✅ 变量选择完成!')
                    
                    st.markdown('<div class="status-box status-success">✅ 变量选择建模完成！</div>', unsafe_allow_html=True)
        
        # 显示结果
        if st.session_state.Adjacency_matrix is not None:
            st.markdown("#### 📊 变量选择结果")
            
            tab4_1, tab4_2, tab4_3 = st.tabs(["模型参数", "选择矩阵", "邻接矩阵"])
            
            with tab4_1:
                st.markdown("##### 🎯 最优参数")
                params_df = pd.DataFrame(st.session_state.best_params_list, 
                                       index=st.session_state.variables_names)
                st.dataframe(params_df, use_container_width=True)
            
            with tab4_2:
                st.markdown("##### 🔍 组选择矩阵")
                selection_df = pd.DataFrame(st.session_state.group_selection_all, 
                                          columns=st.session_state.variables_names,
                                          index=st.session_state.variables_names)
                
                # 使用热力图显示选择矩阵
                fig_selection = px.imshow(selection_df, 
                                        x=selection_df.columns, 
                                        y=selection_df.index,
                                        title="组选择结果热力图",
                                        color_continuous_scale="Blues")
                st.plotly_chart(fig_selection, use_container_width=True)
                
                st.dataframe(selection_df, use_container_width=True)
            
            with tab4_3:
                st.markdown("##### 🌐 邻接矩阵")
                adj_df = pd.DataFrame(st.session_state.Adjacency_matrix, 
                                    columns=st.session_state.variables_names,
                                    index=st.session_state.variables_names)
                
                # 使用热力图显示邻接矩阵
                fig_adj = px.imshow(adj_df, 
                                  x=adj_df.columns, 
                                  y=adj_df.index,
                                  title="邻接矩阵热力图",
                                  color_continuous_scale="RdBu_r")
                st.plotly_chart(fig_adj, use_container_width=True)
                
                st.dataframe(adj_df, use_container_width=True)

with tab5:
    st.markdown('<div class="step-header">📊 步骤5: 结果分析与可视化</div>', unsafe_allow_html=True)
    
    if st.session_state.Adjacency_matrix is None:
        st.markdown('<div class="status-box status-warning">⚠️ 请先完成变量选择以生成分析结果</div>', unsafe_allow_html=True)
    else:
        st.session_state.analysis_complete = True
        
        # 创建子标签页用于不同类型的可视化
        vis_tab1, vis_tab2, vis_tab3, vis_tab4 = st.tabs(["🌐 网络图", "📈 效应分解", "📊 统计图表", "📋 结果总结"])
        
        with vis_tab1:
            st.markdown("#### 🌐 网络拓扑分析")
            
            col1, col2 = st.columns([3, 1])
            
            with col2:
                st.markdown("##### 🎨 网络图参数")
                edge_scale = st.number_input('边粗细缩放', value=0.003, format="%.4f", min_value=0.0001, max_value=0.01)
                vertex_size = st.slider('节点大小', 30, 100, 50)
                vertex_label_size = st.slider('标签字体', 8, 20, 12)
                layout_mode = st.selectbox('布局方式', ['circle', 'fr', 'kk'])
                edge_curved = st.slider('边曲率', 0.0, 1.0, 0.2, 0.05)
            
            with col1:
                def plot_lasso_network(adj_matrix, vertex_names=None, edge_scale=0.003, 
                                     vertex_size=50, vertex_label_size=12, layout_mode='circle', edge_curved=0.2):
                    if vertex_names is None:
                        vertex_names = st.session_state.variables_names
                    
                    g = Graph.Weighted_Adjacency(adj_matrix, mode="directed")
                    g.vs["name"] = vertex_names
                    g.vs["label"] = vertex_names
                    g.simplify(loops=True)

                    layout = g.layout(layout_mode)
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
                        "bbox": (600, 600),
                        "margin": 50,
                        "edge_curved": edge_curved,
                    }

                    out_degrees = g.outdegree()
                    in_degrees = g.indegree()
                    y_pos = np.arange(len(vertex_names))

                    fig = plt.figure(figsize=(12, 8))
                    gs = fig.add_gridspec(1, 2, width_ratios=[3, 1])

                    ax1 = fig.add_subplot(gs[0])
                    plot(g, target=ax1, **visual_style)
                    ax1.set_title('网络拓扑图', fontsize=16, fontweight='bold')
                    ax1.axis('off')

                    ax2 = fig.add_subplot(gs[1])
                    bar_width = 0.6
                    ax2.barh(y_pos, [-d for d in in_degrees], height=bar_width, 
                            color='lightgreen', label='入度', alpha=0.8)
                    ax2.barh(y_pos, out_degrees, height=bar_width, 
                            color='skyblue', label='出度', left=0, alpha=0.8)
                    ax2.axvline(x=0, color='gray', linestyle='--', linewidth=1)
                    ax2.set_yticks(y_pos)
                    ax2.set_yticklabels(vertex_names, fontsize=10)
                    ax2.set_xticks([])
                    ax2.spines['top'].set_visible(False)
                    ax2.spines['right'].set_visible(False)
                    ax2.spines['bottom'].set_visible(False)
                    ax2.spines['left'].set_visible(False)
                    ax2.legend(loc='upper right', fontsize=10, framealpha=0.8)
                    ax2.set_title('度分布', fontsize=12, fontweight='bold')
                    
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
            
            # 网络统计信息
            st.markdown("#### 📈 网络统计信息")
            adj_matrix = st.session_state.Adjacency_matrix
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                total_edges = np.count_nonzero(adj_matrix)
                st.metric("边的数量", total_edges)
            
            with col2:
                positive_edges = np.count_nonzero(adj_matrix > 0)
                st.metric("正向连接", positive_edges)
            
            with col3:
                negative_edges = np.count_nonzero(adj_matrix < 0)
                st.metric("负向连接", negative_edges)
            
            with col4:
                max_weight = np.max(np.abs(adj_matrix))
                st.metric("最大权重", f"{max_weight:.4f}")
        
        with vis_tab2:
            st.markdown("#### 📈 效应分解分析")
            
            if st.session_state.all_effects is not None:
                # 选择要分析的变量
                selected_var = st.selectbox("选择分析变量", st.session_state.variables_names)
                var_idx = list(st.session_state.variables_names).index(selected_var)
                
                # 获取数据
                Y = st.session_state.df.values
                Y = (Y - Y.mean(axis=0)) / Y.std(axis=0)
                effect_data = st.session_state.all_effects[var_idx]
                
                # 创建交互式效应分解图
                fig = go.Figure()
                
                # 添加真实数据
                fig.add_trace(go.Scatter(
                    x=st.session_state.time,
                    y=Y[:, var_idx],
                    mode='markers',
                    name='真实值',
                    marker=dict(color='black', size=4, opacity=0.6)
                ))
                
                # 添加自身效应
                independent_effect = effect_data[:, var_idx]
                if np.any(independent_effect != 0):
                    fig.add_trace(go.Scatter(
                        x=st.session_state.time,
                        y=independent_effect,
                        mode='lines',
                        name=f'{selected_var} 自身效应',
                        line=dict(color='red', width=2)
                    ))
                
                # 添加其他变量的效应
                colors = px.colors.qualitative.Set3
                for j in range(st.session_state.variables_numbers):
                    if j != var_idx:
                        dependent_effect = effect_data[:, j]
                        if np.any(dependent_effect != 0):
                            fig.add_trace(go.Scatter(
                                x=st.session_state.time,
                                y=dependent_effect,
                                mode='lines',
                                name=f'{st.session_state.variables_names[j]} → {selected_var}',
                                line=dict(color=colors[j % len(colors)], width=2)
                            ))
                
                # 添加总效应
                total_effect = effect_data[:, st.session_state.variables_numbers]
                if np.any(total_effect != 0):
                    fig.add_trace(go.Scatter(
                        x=st.session_state.time,
                        y=total_effect,
                        mode='lines',
                        name='总效应',
                        line=dict(color='blue', width=3, dash='dash')
                    ))
                
                fig.update_layout(
                    title=f'{selected_var} 的效应分解',
                    xaxis_title='时间',
                    yaxis_title='效应值',
                    height=500,
                    hovermode='x unified'
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # 效应贡献比例
                st.markdown("##### 📊 效应贡献分析")
                contributions = []
                labels = []
                
                for j in range(st.session_state.variables_numbers):
                    effect_contribution = np.abs(effect_data[:, j]).sum()
                    if effect_contribution > 0:
                        contributions.append(effect_contribution)
                        if j == var_idx:
                            labels.append(f'{st.session_state.variables_names[j]} (自身)')
                        else:
                            labels.append(f'{st.session_state.variables_names[j]}')
                
                if contributions:
                    fig_pie = px.pie(values=contributions, names=labels, 
                                   title=f'{selected_var} 的效应贡献比例')
                    st.plotly_chart(fig_pie, use_container_width=True)
        
        with vis_tab3:
            st.markdown("#### 📊 统计图表分析")
            
            tab_stats1, tab_stats2, tab_stats3 = st.tabs(["数据分布", "相关性分析", "时间序列"])
            
            with tab_stats1:
                # 原始数据vs预处理数据对比
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("##### 原始数据分布")
                    fig_orig = px.box(st.session_state.df, title="原始数据箱线图")
                    fig_orig.update_layout(height=400)
                    st.plotly_chart(fig_orig, use_container_width=True)
                
                with col2:
                    st.markdown("##### 预处理后数据分布")
                    processed_df = pd.DataFrame(st.session_state.Y_tilde, 
                                              columns=st.session_state.variables_names)
                    fig_proc = px.box(processed_df, title="预处理后数据箱线图")
                    fig_proc.update_layout(height=400)
                    st.plotly_chart(fig_proc, use_container_width=True)
            
            with tab_stats2:
                # 相关性分析
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("##### 原始数据相关性")
                    corr_orig = st.session_state.df.corr()
                    fig_corr_orig = px.imshow(corr_orig, title="原始数据相关性",
                                            color_continuous_scale="RdBu_r")
                    fig_corr_orig.update_layout(height=400)
                    st.plotly_chart(fig_corr_orig, use_container_width=True)
                
                with col2:
                    st.markdown("##### 网络连接强度")
                    fig_adj = px.imshow(st.session_state.Adjacency_matrix,
                                      x=st.session_state.variables_names,
                                      y=st.session_state.variables_names,
                                      title="网络邻接矩阵",
                                      color_continuous_scale="RdBu_r")
                    fig_adj.update_layout(height=400)
                    st.plotly_chart(fig_adj, use_container_width=True)
            
            with tab_stats3:
                # 时间序列分析
                st.markdown("##### 多变量时间序列")
                df_reset = st.session_state.df.reset_index()
                fig_ts = px.line(df_reset, x='Time', 
                               y=st.session_state.df.columns.tolist(),
                               title="所有变量的时间序列")
                fig_ts.update_layout(height=500, xaxis_title="时间")
                st.plotly_chart(fig_ts, use_container_width=True)
        
        with vis_tab4:
            st.markdown("#### 📋 分析结果总结")
            
            # 创建结果总结
            st.markdown("##### 🎯 模型性能概览")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("""
                <div class="metric-card">
                    <h4>📊 数据概况</h4>
                    <p>样本数量: {}</p>
                    <p>变量数量: {}</p>
                    <p>特征数量: {}</p>
                </div>
                """.format(
                    st.session_state.time_numbers,
                    st.session_state.variables_numbers,
                    st.session_state.variables_numbers * (n_order + 1)
                ), unsafe_allow_html=True)
            
            with col2:
                total_edges = np.count_nonzero(st.session_state.Adjacency_matrix)
                positive_edges = np.count_nonzero(st.session_state.Adjacency_matrix > 0)
                negative_edges = np.count_nonzero(st.session_state.Adjacency_matrix < 0)
                
                st.markdown("""
                <div class="metric-card">
                    <h4>🌐 网络结构</h4>
                    <p>总连接数: {}</p>
                    <p>正向连接: {}</p>
                    <p>负向连接: {}</p>
                </div>
                """.format(total_edges, positive_edges, negative_edges), unsafe_allow_html=True)
            
            with col3:
                sparsity = 1 - (total_edges / (st.session_state.variables_numbers ** 2))
                max_weight = np.max(np.abs(st.session_state.Adjacency_matrix))
                
                st.markdown("""
                <div class="metric-card">
                    <h4>📈 模型特征</h4>
                    <p>网络稀疏度: {:.2%}</p>
                    <p>最大权重: {:.4f}</p>
                    <p>模型阶数: {}</p>
                </div>
                """.format(sparsity, max_weight, n_order), unsafe_allow_html=True)
            
            # 最优参数表格
            st.markdown("##### 🎯 模型参数总结")
            params_summary_df = pd.DataFrame(st.session_state.best_params_list, 
                                           index=st.session_state.variables_names)
            st.dataframe(params_summary_df, use_container_width=True)
            
            # 变量重要性排序
            st.markdown("##### 🔍 变量重要性分析")
            importance_scores = []
            for i, var_name in enumerate(st.session_state.variables_names):
                # 计算该变量作为预测变量的总重要性（出度）
                out_importance = np.sum(np.abs(st.session_state.Adjacency_matrix[:, i]))
                # 计算该变量作为被预测变量的总重要性（入度）
                in_importance = np.sum(np.abs(st.session_state.Adjacency_matrix[i, :]))
                importance_scores.append({
                    '变量': var_name,
                    '出度重要性': out_importance,
                    '入度重要性': in_importance,
                    '总重要性': out_importance + in_importance
                })
            
            importance_df = pd.DataFrame(importance_scores)
            importance_df = importance_df.sort_values('总重要性', ascending=False)
            st.dataframe(importance_df, use_container_width=True)
            
            # 导出按钮
            st.markdown("##### 💾 结果导出")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("导出邻接矩阵", key="export_adj_matrix"):
                    adj_csv = pd.DataFrame(st.session_state.Adjacency_matrix,
                                         columns=st.session_state.variables_names,
                                         index=st.session_state.variables_names).to_csv()
                    st.download_button("下载邻接矩阵", adj_csv, "adjacency_matrix.csv", "text/csv", key="download_adj_matrix")
            
            with col2:
                if st.button("导出模型参数", key="export_model_params"):
                    params_csv = pd.DataFrame(st.session_state.best_params_list,
                                            index=st.session_state.variables_names).to_csv()
                    st.download_button("下载模型参数", params_csv, "model_parameters.csv", "text/csv", key="download_model_params")
            
            with col3:
                if st.button("导出重要性分析", key="export_importance"):
                    importance_csv = importance_df.to_csv(index=False)
                    st.download_button("下载重要性分析", importance_csv, "variable_importance.csv", "text/csv", key="download_importance")

# 页脚
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 20px;'>
    <p>🧬 idopNetwork 数据分析平台 v0.2 | 
    基于 ASGL 算法的网络推断工具 | 
    <a href='#' style='color: #1f77b4;'>使用说明</a> | 
    <a href='#' style='color: #1f77b4;'>技术支持</a></p>
</div>
""", unsafe_allow_html=True) 
