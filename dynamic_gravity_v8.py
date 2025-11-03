import warnings
from typing import Self

import numpy as np
import pandas as pd
from pandas import DataFrame
from sklearn.preprocessing import MinMaxScaler
from scipy.optimize import minimize
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import re

from AlgoCaller.algokit import BaseAlgorithm, MetricMenu

warnings.filterwarnings('ignore')

# Done 更改：
# 1. 更改固定列名
# 2. 更改聚合写入，删除冗余部分
# 3. numpy处理有效距离的计算
# 4. 解决theta_params的问题（指标参数数量、显示为0.的情况）并测试指标输入 结果正常
# 5. 解决数据输入问题（只需要输入指标即可 不需要有对照_i或_j的处理）
# 6. 优化infer的向量化  
# 7. 规定用户输入的属性限定在节点属性（md文档中说明）
# 8. 添加参数外推部分的模式（random_walk、linear、ar1）
# 9. 熵权法被替换为PCA方法对指标进行处理
# 10. 精简数据处理函数的调用次数
# 11. 完成用户任意输入指标列名的处理
# 12. 降低数据传输中的冗余



class DynamicGravityIndicator(BaseAlgorithm):

    AlgoName = "动态重力指标模型"
    AlgoVersion = "2.0.0"
    AlgoRemarks = {"tunable": False, "cross_validation": False}
    Hyperparam_Names = (
        "use_effective_distance",
        "fill_missing",
        "mode",
        "n_components"
    )

    Metrics = ("R2", "MSE", "RMSE", "MAE", "MAPE", "SMAPE")

    def __init__(
        self,
        use_effective_distance: bool = True,
        fill_method: str = "mean",
        mode: str = "random_walk",
        n_components: int = 2,
    ):
        super().__init__(use_effective_distance=use_effective_distance, fill_method=fill_method, mode=mode, n_components=n_components)

        self.use_effective_distance = use_effective_distance  # 有效距离是否加入 True就加入
        self.fill_method = fill_method
        self.n_components = n_components
        self.mode = mode # 参数估计方法的选择
        self.len_indicators = 0 # 输入指标数量
        self.last_year = None
        self.city_pairs = None
        self.theta_params = {}
        self.all_cities = []  # 存储所有城市节点
        self.n_cities = 0

    def _build_complete_matrix(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        构建完整的n×n城市矩阵，包含自环(i=j)
        将指标放入到列表attr_i以及attr_j中
        """
        # 1. 获取城市节点
        self.all_cities = np.unique(pd.concat([data.iloc[:, 1], data.iloc[:, 2]])).tolist()
        self.n_cities = len(self.all_cities)

        # 2. 构建年份 x 城市对的笛卡尔积索引
        year_uni = np.unique(data.iloc[:, 0])
        attr_cols = data.columns.tolist()[:3]
        idx = pd.MultiIndex.from_product([year_uni, self.all_cities, self.all_cities], names=attr_cols)
        df = pd.DataFrame(index=idx).reset_index().merge(data, on=attr_cols, how="left")

        # 3. 拆分基础列与节点属性列
        base_cols = ['year', 'Vi', 'Vj', 'Tij']
        node_cols = df.columns.tolist()[4:]  # 指标列
        self.len_indicators = len(data.columns) - len(base_cols)
        df.columns = base_cols + node_cols

        # 4. 缺失值处理
        df = self._missing_process(df)
        
        # 5. 构建城市属性查找表
        city_attr = (
            pd.concat([
                df[['year', 'Vi'] + node_cols].rename(columns={'Vi': 'city'}),
                df[['year', 'Vj'] + node_cols].rename(columns={'Vj': 'city'})
            ])
            .drop_duplicates(subset=['year', 'city'])
            .sort_values(['year', 'city'])
        )
        
        # 6. 创建最终结果
        result = df[base_cols].copy()
        
        # 添加attr_i（Vi对应的指标列表）
        result = result.merge(
            city_attr.rename(columns={'city': 'Vi'}),
            on=['year', 'Vi'],
            how='left'
        )
        result['attr_i'] = result[node_cols].values.tolist()
        
        # 添加attr_j（Vj对应的指标列表）
        result = result.merge(
            city_attr.rename(columns={'city': 'Vj'}),
            on=['year', 'Vj'],
            how='left',
            suffixes=('', '_j')
        )
        result['attr_j'] = result[[f'{col}_j' for col in node_cols]].values.tolist()
        
        # 7. 只保留最终需要的列
        final_cols = ['year', 'Vi', 'Vj', 'Tij', 'attr_i', 'attr_j']
        result = result[final_cols]
        
        # print(result)
        # 1     2020  乐山市        内江市       68.0  [2003.43, 3452593.0, 68960.4608912591]  [1465.88, 4032061.0, 67572.8445960943]
        return result

    def _missing_process(self, df: DataFrame) -> DataFrame:
        """
        用于处理缺失值
        """
        df_missing_process = df.copy()

        # 情况1 城市流动量缺失
        flow_missing = df_missing_process[df_missing_process['Tij'].isna()]

        if not flow_missing.empty:
            missing_ratio = len(flow_missing) / len(df_missing_process)
            if missing_ratio > 0.5:
                raise ValueError("流动量缺失值过多（超过50%）,请检查输入数据")
            if self.fill_method == "zero":
                df_missing_process['Tij'] = df_missing_process['Tij'].fillna(0)

            elif self.fill_method == "mean":
                # 分组求取平均来填充缺失的流动量
                df_missing_process['Tij'] = df_missing_process.groupby(['Vi', 'Vj'])['Tij'].transform(
                    lambda x: x.fillna(x.mean())
                )

        # 情况2 指标缺失
        other_ind = df_missing_process[df_missing_process.iloc[:, 4:].isna().any(axis=1)]
        if not other_ind.empty:
            raise ValueError("输入的数据中存在指标缺失，请检查输入数据")

        return df_missing_process

    def _pca_indicators(self, data: pd.DataFrame, indicator_columns: list) -> pd.DataFrame:
        """
        PCA处理数据 输入指标数量至少大于两个
        且通过线性平移的方式来确保PCA结果为正值 否则会在参数评估地方出现nan情况
        """
        # 前四列
        result_data = data.iloc[:, :4].copy()

        if self.len_indicators >= 2:
            # 将attr_i和attr_j的列表转换为numpy数组
            i_data = np.array(data['attr_i'].tolist())  # shape: (n_samples, n_indicators)
            j_data = np.array(data['attr_j'].tolist())  # shape: (n_samples, n_indicators)
            
            scaler = StandardScaler()
            i_scaled = scaler.fit_transform(i_data)
            j_scaled = scaler.transform(j_data)

            pca = PCA(n_components=self.n_components)
            composite_i = pca.fit_transform(i_scaled)
            composite_j = pca.transform(j_scaled)

            # 线性平移确保PCA结果为正值
            # 找到所有PCA分量的最小值
            all_composite = np.vstack([composite_i, composite_j])
            min_val = np.min(all_composite)
            
            # 如果最小值小于等于0，进行平移
            if min_val <= 0:
                shift = abs(min_val) + 1e-8  # 确保所有值为正
                composite_i = composite_i + shift
                composite_j = composite_j + shift
                # print(f"PCA结果平移: 最小值={min_val:.6f}, 平移量={shift:.6f}")
            
            # 一次性生成列名
            dims = np.arange(1, self.n_components + 1)
            i_cols = [f'composite_i_{d}' for d in dims]
            j_cols = [f'composite_j_{d}' for d in dims]

            # 拼接结果
            composite_df = pd.DataFrame(
                np.hstack([composite_i, composite_j]),
                columns=i_cols + j_cols,
                index=data.index
            )

            result_data = pd.concat([result_data, composite_df], axis=1)

            return result_data, composite_df.columns.tolist()

        elif self.len_indicators == 1:
            # 这种需要将指标拆开重命名
            result_data['indicator_i_1'] = data['attr_i'].apply(lambda x: x[0] if isinstance(x, list) and len(x) > 0 else np.nan)
            result_data['indicator_j_1'] = data['attr_j'].apply(lambda x: x[0] if isinstance(x, list) and len(x) > 0 else np.nan)
            processed_columns = ['indicator_i_1', 'indicator_j_1']

            return result_data, processed_columns

        elif self.len_indicators == 0:
            raise ValueError("未识别指标输入")

    def _lag_indicators(self, data: pd.DataFrame, indicator_cols: list) -> pd.DataFrame:
        """对指标进行滞后一期处理"""
        lagged_data = data.copy()
        lagged_data = lagged_data.sort_values('year').reset_index(drop=True)
        # print(lagged_data)
        for col in indicator_cols:
            lagged_col_name = f'{col}_lag1'
            lagged_data[lagged_col_name] = lagged_data.groupby(['Vi', 'Vj'])[col].shift(1)

        return lagged_data

    def _calculate_reverse_effective_distance(self, data: pd.DataFrame, Tij_col: str = 'Tij_lag1') -> np.ndarray:
        """
        计算有效距离（i->j），支持自环情况,
        这里对应的拟合参数应该要小于零,整体成反比"""
        # 构建 flow_matrix
        flow_matrix = data.pivot_table(index='Vi', columns='Vj', values=Tij_col, aggfunc='sum', fill_value=0)
        
        # 每个 i 的总流出量
        i_total_outflow = flow_matrix.sum(axis=1).to_numpy()
        
        # 构建 Vi、Vj 的索引映射
        vi_idx = {v: i for i, v in enumerate(flow_matrix.index)}
        vj_idx = {v: j for j, v in enumerate(flow_matrix.columns)}
        
        # 将 Vi、Vj 转换为矩阵索引
        vi_array = data['Vi'].map(vi_idx).to_numpy()
        vj_array = data['Vj'].map(vj_idx).to_numpy()
        
        # 获取对应的 Tij 值（向量化）
        Tij_array = flow_matrix.to_numpy()[vi_array, vj_array]
        
        # 对应 i 的总流出量
        total_outflow_array = i_total_outflow[vi_array]
        
        # 计算概率并防止除零
        with np.errstate(divide='ignore', invalid='ignore'):
            p_ij = np.divide(Tij_array, total_outflow_array, out=np.zeros_like(Tij_array), where=total_outflow_array != 0)
        
        # 防止 log(0)
        p_ij = np.where(p_ij == 0, 1e-10, p_ij)
        
        # 计算 effD_ij
        effD_ij = 1 - np.log(p_ij)
        
        return effD_ij

    def _calculate_Aji_with_parameters(self, data: pd.DataFrame) -> np.ndarray:
        """
        计算带参数的Aji，支持自环情况 - 与_fit_parameters对齐的向量化实现
        """
        # 识别指标列
        i_cols = data.filter(regex=r'_i_\d+_lag1$').columns.tolist()
        j_cols = data.filter(regex=r'_j_\d+_lag1$').columns.tolist()

        # 判断是否为自环
        self_loop_mask = data['Vi'].values == data['Vj'].values

        # 初始化指标部分
        if i_cols and j_cols:
            Xi = data[i_cols].values
            Xj = data[j_cols].values
            
            # 与_fit_parameters完全一致的log-ratio处理
            X_ratio = np.log(Xj / (Xi + 1e-10) + 1e-10)

            K = X_ratio.shape[1]  # 特征数量
            # 获取参数 - 现在使用与_fit_parameters相同的参数结构
            # 假设theta_params中存储的是所有特征的theta向量
            theta = self.theta_params.get('theta_ratio', np.ones(K))
            
            # 向量化计算：与_fit_parameters中的 U_flat = X_ratio @ theta 保持一致
            indicator_part = np.exp(X_ratio @ theta)
            
            # 自环情况特殊处理：指标部分为1
            indicator_part[self_loop_mask] = 1.0
        else:
            indicator_part = np.ones(len(data))

        # 计算有效距离部分 - 与_fit_parameters中的 gamma * (effD - 1) 对应
        if self.use_effective_distance and 'effD_ij' in data.columns:
            gamma = self.theta_params.get('gamma', 1.0)
            effD = data['effD_ij'].values + 1e-10
            # 注意：这里与拟合函数中的 -gamma * (effD - 1) 对应
            # 在概率计算中是指数形式：exp(-gamma * (effD - 1))
            distance_part = np.exp(-gamma * (effD - 1))
        else:
            distance_part = 1.0

        return indicator_part * distance_part

    def _process_training_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """处理训练数据，构建完整矩阵,输入原始数据"""

        # 构建完整的n×n矩阵
        complete_data = self._build_complete_matrix(data)

        # 识别自定义指标 attr_i和attr_j
        custom_indicators = [col for col in complete_data.columns[4:]]

        if not custom_indicators:
            raise ValueError("未找到自定义指标列")

        # 处理指标 判断指标数量
        if self.n_components <= self.len_indicators:
            # 执行PCA逻辑
            result_data, agg_indicators_cols = self._pca_indicators(complete_data, custom_indicators)
        else:
            # 此处不执行PCA
            attr_i_df = pd.DataFrame(complete_data['attr_i'].tolist(),index=complete_data.index,columns=[f'indicator_i_{k+1}' for k in range(self.len_indicators)])
            attr_j_df = pd.DataFrame(complete_data['attr_j'].tolist(),index=complete_data.index,columns=[f'indicator_j_{k+1}' for k in range(self.len_indicators)])

            result_data = pd.concat([complete_data,attr_i_df,attr_j_df],axis=1)
            agg_indicators_cols = attr_i_df.columns.tolist() + attr_j_df.columns.tolist()

        # 构建基础数据 + 处理完成的数据
        # 聚合指标还要再加上流动量做滞后性指标
        lagged_cols = agg_indicators_cols + ['Tij']
        # 滞后处理数据
        lagged_data = self._lag_indicators(result_data, lagged_cols)

        # 有效距离计算
        if self.use_effective_distance:
            lagged_data['effD_ij'] = self._calculate_reverse_effective_distance(lagged_data, 'Tij_lag1')

        final_data = lagged_data.dropna().reset_index(drop=True)

        # 最后列名包含基础数据+处理后的指标的滞后数据+Tij的滞后数据+滞后的有效距离
        # print(final_data.columns.tolist())
        return final_data

    def _fit_parameters(self, data: pd.DataFrame):
        """
        基于最大似然估计（MLE）拟合参数，向量化实现
        - 支持自环
        
        """
        # 识别指标列
        i_cols = data.filter(regex=r'_i_\d+_lag1$').columns.tolist()
        j_cols = data.filter(regex=r'_j_\d+_lag1$').columns.tolist()

        years = sorted(data['year'].unique())
        theta_history = {}

        # 计算每年的参数
        for t in years:
            subset = data[data['year'] == t]
            Tij = subset['Tij'].values + 1e-10  # 避免 log(0)
            effD = subset['effD_ij'].values + 1e-10

            Xi = subset[i_cols].values
            Xj = subset[j_cols].values

            # log-ratio 特征矩阵
            X_ratio = np.log(Xj / (Xi + 1e-10) + 1e-10)
            # print(X_ratio.shape)

            K = X_ratio.shape[1]  # 特征数量
            
            # 定义负对数似然函数（现在包含完整的参数定义）
            def neg_log_likelihood(params):
                theta = params[:K]
                gamma = params[K]
                
                # 计算一维效用值
                U_flat = X_ratio @ theta - gamma * (effD - 1)
                
                #  n x n 矩阵
                n = int(np.sqrt(len(U_flat)))
                U_matrix = U_flat.reshape(n, n)

                # 矩阵计算 softmax
                U_max = np.max(U_matrix, axis=1, keepdims=True)
                exp_U = np.exp(U_matrix - U_max)
                sum_exp_U = np.sum(exp_U, axis=1, keepdims=True)
                P_matrix = exp_U / (sum_exp_U + 1e-12)
                # print(P_matrix)
                # 重塑 Tij
                Tij_matrix = Tij.reshape(n, n)
                
                return -np.sum(Tij_matrix * np.log(P_matrix + 1e-12))
            
            # 初始化参数（K个theta + 1个gamma）
            initial_params = np.zeros(K + 1)
            
            # 优化
            result = minimize(neg_log_likelihood, initial_params, method='L-BFGS-B')

            # 初始参数
            init_params = np.concatenate([
                np.random.uniform(0.01, 0.1, K), [0.01]])

            # 优化求解
            bounds = [(-1, None)] * (K + 1)  # θ_k 可大于0或者小于0, γ ≥ 0

            try:
                result = minimize(neg_log_likelihood, init_params, method='L-BFGS-B', bounds=bounds,
                options={'ftol': 1e-8, 'maxiter': 1000})

                if result.success:
                    theta_est = result.x[:K]
                    gamma_est = result.x[-1]
                
                    theta_history[t] = {
                        'theta': theta_est,
                        'gamma': gamma_est,
                    }
                else:
                    raise ValueError(f"警告: 年份 {t} 优化失败: {result.message}")
            except Exception as e:
                raise ValueError(f"错误: 年份 {t} 优化异常: {e}")

            # 保存每年估计结果
            theta_history[t] = {
                'theta': theta_est,
                'gamma': gamma_est,
            }

        # print(theta_history)
        # 设置最新年份参数
        if theta_history:
            last_t = max(theta_history.keys())
            self.theta_params = theta_history[last_t]
            
        else:
            raise ValueError("未成功估计任何年份参数。")

        return theta_history

    def _forecast_parameters(self, data: pd.DataFrame):
        """
        参数预测模式:
        'linear': 线性趋势外推
        'random_walk': 随机游走预测
        """
        theta_history = self._fit_parameters(self.trained_data)
        years = sorted(theta_history.keys())
        T = len(years)
        if T < 2:
            raise ValueError("数据年份不足，无法进行趋势外推。")

        # 构造时间序列
        theta_matrix = np.array([theta_history[y]['theta'] for y in years])  # shape (T, K) 时间外推，指标参数
        # print(theta_matrix.shape)
        gamma_series = np.array([theta_history[y]['gamma'] for y in years])  # shape (T,)

        if self.mode == 'linear':
            x = np.arange(T)
            # 向量化线性拟合 theta
            # 对每列 theta_k 使用最小二乘法计算 slope 和 intercept
            X_design = np.vstack([x, np.ones(T)]).T  # shape (T, 2)
            # print(X_design.shape)
            # 每列独立计算: θ_k = slope_k * x + intercept_k
            theta_next = np.linalg.lstsq(X_design, theta_matrix, rcond=None)[0]  # shape (2, K)
            theta_pred = theta_next[0, :] * (T) + theta_next[1, :]  # 下一期预测
            # gamma 线性拟合（标量）
            slope_g, intercept_g = np.linalg.lstsq(X_design, gamma_series, rcond=None)[0]
            gamma_pred = slope_g * (T) + intercept_g

        elif self.mode == 'random_walk':
            delta_theta = np.diff(theta_matrix, axis=0)
            delta_gamma = np.diff(gamma_series)
            mu_theta = delta_theta.mean(axis=0)
            mu_gamma = delta_gamma.mean()
            theta_pred = theta_matrix[-1, :] + mu_theta
            gamma_pred = gamma_series[-1] + mu_gamma
        
        elif self.mode == 'ar1':
            theta_pred = np.zeros(theta_matrix.shape[1])

            for k in range(theta_matrix.shape[1]):
                y = theta_matrix[:, k]
                y_lag = y[:-1]
                y_next = y[1:]
                X = np.vstack([y_lag, np.ones_like(y_lag)]).T
                rho, c = np.linalg.lstsq(X, y_next, rcond=None)[0]  # y_t = ρ*y_{t-1} + c
                theta_pred[k] = rho * y[-1] + c  # 下一期预测

            # gamma 的 AR(1) 拟合
            g_lag = gamma_series[:-1]
            g_next = gamma_series[1:]
            Xg = np.vstack([g_lag, np.ones_like(g_lag)]).T
            rho_g, c_g = np.linalg.lstsq(Xg, g_next, rcond=None)[0]
            gamma_pred = rho_g * gamma_series[-1] + c_g
        else:
            raise ValueError("当前模式仅支持linear 或random walk")

        # 更新参数
        self.theta_params = {
            'theta': theta_pred,
            'gamma': gamma_pred,
        }

    def train(self, features: pd.DataFrame) -> Self:
        """训练功能"""
        # 处理数据
        pattern = re.compile(r'^(composite|indicator)_[ij]_\d+$')
        processed_data = self._process_training_data(features)
        filter_processd_data = processed_data.loc[:, ~processed_data.columns.str.match(pattern)]
        # print(processed_data.columns.tolist())
        # 拟合参数
        self._fit_parameters(filter_processd_data)
        # print(processed_data.columns.tolist())
        
        # 使用参数重新计算Aji
        filter_processd_data['Aji'] = self._calculate_Aji_with_parameters(filter_processd_data)
        filter_processd_data['effD_ij'] = filter_processd_data.groupby('Vj')['Aji'].transform(lambda x: x / x.sum())

        # 计算训练数据的拟合值
        filter_processd_data['Tij_fitted'] = filter_processd_data.groupby('Vi')['Tij_lag1'].transform('sum') * filter_processd_data['effD_ij']
        # 计算训练评估指标
        y_true_train = filter_processd_data['Tij'].values
        y_pred_train = filter_processd_data['Tij_fitted'].values
        self.metrics = {m: MetricMenu.create(m, data=(y_true_train, y_pred_train)) for m in self.Metrics}

        # 保存训练状态
        self.trained_data = filter_processd_data
        # print(self.trained_data.columns.tolist())
        self.last_year = filter_processd_data['year'].max()
        self.city_pairs = filter_processd_data[['Vi', 'Vj']].drop_duplicates().reset_index(drop=True)
        return self

    def infer(self, steps: int = 1) -> pd.DataFrame:
        """预测功能，输出完整的n×n流动量"""
        # 获取最新的完整数据
        latest_data = self.trained_data[self.trained_data['year'] == self.last_year].copy()
        forecast_results = []
        current_year = self.last_year

        for step in range(1, steps + 1):
            forecast_year = current_year + step

            # 外推参数
            self._forecast_parameters(self.trained_data)

            # 预测当前期的Aji
            latest_data['year'] = forecast_year
            latest_data['Aji'] = self._calculate_Aji_with_parameters(latest_data)
            latest_data['effD_ij'] = latest_data.groupby('Vj')['Aji'].transform(lambda x: x / x.sum())

            # 计算Fij (预测的Tij)
            Ti_total = latest_data.groupby('Vi')['Tij_lag1'].sum().to_dict()
            latest_data['Ti_total'] = latest_data['Vi'].map(Ti_total)
            latest_data['Fij'] = latest_data['Ti_total'] * latest_data['effD_ij']

            # 为下一步更新Tij_lag1
            latest_data['Tij_lag1'] = latest_data['Fij'].values

            # 保存结果
            forecast_step = latest_data[['year', 'Vi', 'Vj', 'Fij', 'effD_ij', 'Aji']].copy()
            forecast_step.rename(columns={'Fij': 'Tij_forecast'}, inplace=True)
            forecast_results.append(forecast_step)

        result = pd.concat(forecast_results, ignore_index=True)

        # 保存较为完整的数据
        all_cities = self.all_cities
        
        # 构建年份 × 城市 × 城市 的笛卡尔积
        years = result['year'].unique()
        idx = pd.MultiIndex.from_product([years, all_cities, all_cities], names=['year', 'Vi', 'Vj'])
        complete_df = pd.DataFrame(index=idx).reset_index()
        
        # merge 原始预测结果
        complete_result = complete_df.merge(result, on=['year', 'Vi', 'Vj'], how='left')
        
        # 填充缺失值
        fill_cols = ['Tij_forecast', 'effDij_lag1', 'Aji']
        for col in fill_cols:
            if col in complete_result.columns:
                complete_result[col] = complete_result[col].fillna(0)
        
        # 取前四列
        final_result = complete_result.iloc[:, :4]

        return final_result

    def output(self) -> dict:
        """训练结果输出"""
        return {
            "city_count": len(self.all_cities) if self.all_cities else 0,
            "theta_params": self.theta_params,
            "training_metrics": self.get_metrics()
        }

if __name__ == "__main__":
    # 测试代码
    filepath = r"E:\三合力通\算法商店\算法开发\Example\test_data_single_Pop_GDP_Wage.xlsx"

    data = pd.read_excel(filepath)
    # 需要注意component的个数
    model = DynamicGravityIndicator(n_components=4).train(data)

    print(model.infer(4))
    
    print("评估结果:", model.output())