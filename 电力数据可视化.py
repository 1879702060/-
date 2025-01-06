import pandas as pd
import numpy as np
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import networkx as nx
def load_data(file_path):
    df = pd.read_csv(file_path)
    # 检查 NaN 并填充为 0
    df.fillna(0, inplace=True)
    # 确保数值列是数字类型
    for col in df.columns[3:]:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    # 打印列信息和特征检查
    print("Columns:", df.columns)
    print("Feature Distribution:\n", df['Features'].value_counts())
    return df


# 热力图
def draw_global_heatmap(data, year, feature):
    if feature not in ["net generation", "net consumption"]:
        st.warning(f"Heatmap is available for 'net generation' and 'net consumption'.")
        return

    data_filtered = data[data['Features'] == feature][['Country', year]].rename(columns={year: 'value'})

    fig = px.choropleth(
        data_filtered,
        locations="Country",
        locationmode="country names",
        color="value",
        title=f"{feature} in {year}",
        color_continuous_scale="Reds",
        hover_name="Country"
    )
    fig.update_geos(showland=True, landcolor="white", showocean=True, oceancolor="lightblue")
    fig.update_layout(margin={"r": 0, "t": 40, "l": 0, "b": 0}, geo=dict(showcoastlines=True, coastlinecolor="black", projection_type="natural earth"))
    st.plotly_chart(fig)


# 折线图
def draw_combined_line_plots(data):
    features = ["net generation", "net consumption"]
    year_columns = [col for col in data.columns if col.isdigit()]
    year_columns_numeric = [int(year) for year in year_columns]

    for feature in features:
        # 筛选特定特征数据
        data_filtered = data[data['Features'] == feature]
        if data_filtered.empty:
            st.warning(f"No data available for the feature: {feature}.")
            continue

        # 按区域分组并聚合年份数据
        data_region = data_filtered.groupby('Region')[year_columns].sum()
        if data_region.empty or data_region.sum().sum() == 0:
            st.warning(f"No valid data available for the feature: {feature}.")
            continue

        # 绘制折线图
        plt.figure(figsize=(12, 8))
        for region in data_region.index:
            region_data = data_region.loc[region]
            if region_data.sum() > 0:
                plt.plot(year_columns_numeric, region_data.values, label=region)

        # 图表设置
        plt.legend(loc='upper left', title="Region")
        plt.title(f"Trend of {feature} by Region (1980-2021)", fontsize=16)
        plt.xlabel("Year", fontsize=12)
        plt.ylabel("Value", fontsize=12)
        plt.xticks(year_columns_numeric[::5], rotation=45)
        plt.grid(True)
        plt.tight_layout()

        st.pyplot(plt)

# 绘制气泡图
def draw_bubble_chart(data, year, feature):
    if feature not in ["net generation", "net consumption"]:
        st.warning(f"Bubble chart is available for 'net generation' and 'net consumption'.")
        return

    data_filtered = data[data['Features'] == feature][['Country', 'Region', year]].rename(columns={year: 'value'})

    fig = px.scatter(
        data_filtered,
        x="Country",
        y="Region",
        size="value",  # 气泡大小
        color="value",  # 气泡颜色
        hover_name="Country",
        title=f"Bubble Chart for {feature} in {year}",
        labels={"value": feature},
        size_max=60,
        color_continuous_scale="Viridis"
    )
    fig.update_layout(margin={"r": 0, "t": 40, "l": 0, "b": 0})
    st.plotly_chart(fig)


# 绘制时间序列动画
def draw_time_series_animation(data, feature):
    if feature not in ["net generation", "net consumption"]:
        st.warning(f"Time series animation is available for 'net generation' and 'net consumption'.")
        return

    year_columns = [col for col in data.columns if col.isdigit()]
    data_filtered = data[data['Features'] == feature][['Country', 'Region'] + year_columns]
    data_melted = pd.melt(data_filtered, id_vars=['Country', 'Region'], var_name='Year', value_name='Value')
    data_melted['Year'] = data_melted['Year'].astype(int)

    fig = px.scatter(
        data_melted,
        x="Country",
        y="Value",
        animation_frame="Year",
        animation_group="Country",
        size="Value",
        color="Region",
        hover_name="Country",
        title=f"Time Series Animation for {feature}",
        labels={"Value": feature},
        size_max=60,
    )
    fig.update_layout(margin={"r": 0, "t": 40, "l": 0, "b": 0})
    st.plotly_chart(fig)
# 绘制动态关系图
def draw_network_diagram(data, year, feature):
    if feature not in ["net generation", "net consumption"]:
        st.warning(f"Network diagram is available for 'net generation' and 'net consumption'.")
        return

    data_filtered = data[data['Features'] == feature][['Region', year]].rename(columns={year: 'value'})
    regions = data_filtered['Region'].unique()

    # 构建图
    G = nx.Graph()
    node_values = {}  # 用于存储每个节点的数值

    for region in regions:
        G.add_node(region)
        node_values[region] = data_filtered[data_filtered['Region'] == region]['value'].sum()

    # 计算最大最小值用于颜色归一化
    min_value = min(node_values.values())
    max_value = max(node_values.values())

    # 为简化逻辑，随机模拟一些关系和边权重
    for i, region1 in enumerate(regions):
        for j, region2 in enumerate(regions):
            if i != j:
                weight = (node_values[region1] + node_values[region2]) / 2
                G.add_edge(region1, region2, weight=weight)

    pos = nx.spring_layout(G)

    # 处理边
    edge_x = []
    edge_y = []
    for edge in G.edges(data=True):
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.append(x0)
        edge_x.append(x1)
        edge_x.append(None)
        edge_y.append(y0)
        edge_y.append(y1)
        edge_y.append(None)

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines')

    # 处理节点
    node_x = []
    node_y = []
    text = []
    node_color = []

    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        text.append(f"{node}: {node_values[node]:,.0f}")  # 显示数值
        normalized_value = (node_values[node] - min_value) / (max_value - min_value)  # 归一化
        node_color.append(normalized_value)  # 赋值颜色映射

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        hoverinfo='text',
        marker=dict(
            showscale=True,
            colorscale='YlGnBu',
            reversescale=True,
            color=node_color,  # ✅ 映射颜色
            size=[10 + 20 * v for v in node_color],  # ✅ 大小动态调整
            colorbar=dict(
                thickness=15,
                title='Node Connections',
                xanchor='left',
                titleside='right'
            )),
        text=text)

    fig = go.Figure(data=[edge_trace, node_trace],
                    layout=go.Layout(
                        title=f'Network Diagram for {feature} in {year}',
                        titlefont_size=16,
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=0, l=0, r=0, t=40),
                        xaxis=dict(showgrid=False, zeroline=False),
                        yaxis=dict(showgrid=False, zeroline=False)))

    st.plotly_chart(fig)



# 绘制多层次环形图
def draw_sunburst_chart(data, feature):
    if feature not in ["net generation", "net consumption"]:
        st.warning(f"Sunburst chart is available for 'net generation' and 'net consumption'.")
        return

    year_columns = [col for col in data.columns if col.isdigit()]
    data_filtered = data[data['Features'] == feature]
    data_region = data_filtered.groupby('Region')[year_columns].sum().reset_index()

    # 将数据重构为适合 Sunburst 图的格式
    sunburst_data = []
    for _, row in data_region.iterrows():
        region = row['Region']
        for year in year_columns:
            sunburst_data.append({
                'Region': region,
                'Year': year,
                'Value': row[year]
            })

    df_sunburst = pd.DataFrame(sunburst_data)

    # 绘制 Sunburst 图
    fig = px.sunburst(
        df_sunburst,
        path=['Region', 'Year'],
        values='Value',
        title=f"Sunburst Chart for {feature}",
        color='Value',
        color_continuous_scale='RdYlBu'
    )
    st.plotly_chart(fig)

#地理动态时间热力图
def draw_dynamic_map(data, feature):
    if feature not in ["net generation", "net consumption"]:
        st.warning(f"Dynamic map is available for 'net generation' and 'net consumption'.")
        return

    year_columns = [col for col in data.columns if col.isdigit()]
    data_filtered = data[data['Features'] == feature][['Country'] + year_columns]
    data_melted = pd.melt(data_filtered, id_vars='Country', var_name='Year', value_name='Value')
    data_melted['Year'] = data_melted['Year'].astype(int)

    fig = px.choropleth(
        data_melted,
        locations="Country",
        locationmode="country names",
        color="Value",
        animation_frame="Year",
        title=f"Dynamic Map for {feature}",
        color_continuous_scale="Blues",
        hover_name="Country"
    )
    st.plotly_chart(fig)

#雷达图
def draw_radar_chart(data, year):
    year_column = str(year)
    data_filtered = data[data['Features'] == "net generation"].groupby('Region')[year_column].sum().reset_index()

    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=data_filtered[year_column],
        theta=data_filtered['Region'],
        fill='toself',
        name=f"Net Generation in {year}"
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True)
        ),
        title=f"Radar Chart for Net Generation ({year})"
    )
    st.plotly_chart(fig)

#堆积面积图
def draw_stacked_area_chart(data, feature):
    if feature not in ["net generation", "net consumption"]:
        st.warning(f"Stacked area chart is available for 'net generation' and 'net consumption'.")
        return

    year_columns = [col for col in data.columns if col.isdigit()]
    data_filtered = data[data['Features'] == feature]
    data_grouped = data_filtered.groupby('Region')[year_columns].sum().T  # 转置为年份行索引

    fig = px.area(
        data_grouped,
        x=data_grouped.index,
        y=data_grouped.columns,
        title=f"Stacked Area Chart for {feature}",
        labels={'value': 'Value', 'index': 'Year'},
        color_discrete_sequence=px.colors.qualitative.Set3
    )
    st.plotly_chart(fig)

def draw_sankey_chart(data, feature, year):
    if feature not in ["net generation", "net consumption"]:
        st.warning(f"Sankey diagram is available for 'net generation' and 'net consumption'.")
        return

    data_filtered = data[data['Features'] == feature][['Region', 'Country', str(year)]].rename(columns={str(year): 'Value'})
    data_filtered = data_filtered[data_filtered['Value'] > 0]  # 移除值为0的数据

    # 桑基图节点和链接
    regions = data_filtered['Region'].unique()
    countries = data_filtered['Country'].unique()

    node_labels = list(regions) + list(countries)
    node_indices = {label: idx for idx, label in enumerate(node_labels)}

    links = {
        "source": [node_indices[region] for region in data_filtered['Region']],
        "target": [node_indices[country] for country in data_filtered['Country']],
        "value": data_filtered['Value'].tolist(),
    }

    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=node_labels
        ),
        link=links
    )])

    fig.update_layout(title_text=f"Sankey Diagram for {feature} in {year}", font_size=10)
    st.plotly_chart(fig)

#热力属兔
def draw_treemap(data, feature, year):
    if feature not in ["net generation", "net consumption"]:
        st.warning(f"Treemap is available for 'net generation' and 'net consumption'.")
        return

    data_filtered = data[data['Features'] == feature][['Region', 'Country', str(year)]].rename(columns={str(year): 'Value'})
    data_filtered = data_filtered[data_filtered['Value'] > 0]  # 过滤值为0的数据

    fig = px.treemap(
        data_filtered,
        path=["Region", "Country"],  # 层级关系：区域 -> 国家
        values="Value",
        color="Value",
        color_continuous_scale="Viridis",
        title=f"Treemap for {feature} in {year}",
        labels={"Value": feature}
    )
    st.plotly_chart(fig)
# 平行坐标图
def draw_parallel_coordinates(data, feature):
    if feature not in ["net generation", "net consumption"]:
        st.warning(f"Parallel coordinates plot is available for 'net generation' and 'net consumption'.")
        return

    year_columns = [col for col in data.columns if col.isdigit()]
    data_filtered = data[data['Features'] == feature][['Country', 'Region'] + year_columns]
    data_filtered = data_filtered[data_filtered[year_columns].sum(axis=1) > 0]  # 过滤全为0的数据

    fig = px.parallel_coordinates(
        data_filtered,
        dimensions=year_columns,  # 使用年份列作为维度
        color=data_filtered[year_columns].mean(axis=1),  # 用平均值作为颜色
        color_continuous_scale="Viridis",
        title=f"Parallel Coordinates Plot for {feature}"
    )
    st.plotly_chart(fig)

# 径向柱状图
def draw_radial_bar_chart(data, feature, year):
    if feature not in ["net generation", "net consumption"]:
        st.warning(f"Radial bar chart is available for 'net generation' and 'net consumption'.")
        return

    data_filtered = data[data['Features'] == feature][['Region', str(year)]].rename(columns={str(year): 'Value'})
    data_filtered = data_filtered.groupby('Region').sum().reset_index()

    fig = px.bar_polar(
        data_filtered,
        r='Value',
        theta='Region',
        color='Value',
        color_continuous_scale="Blues",
        title=f"Radial Bar Chart for {feature} in {year}",
        labels={'Value': feature}
    )
    st.plotly_chart(fig)

# 甘特图 (Gantt Chart)
def draw_gantt_chart(data, feature):
    if feature not in ["net generation", "net consumption"]:
        st.warning(f"Gantt chart is only available for 'net generation' and 'net consumption'.")
        return

    year_columns = [col for col in data.columns if col.isdigit()]
    data_filtered = data[data['Features'] == feature][['Country', 'Region'] + year_columns]
    data_filtered = data_filtered.melt(id_vars=['Country', 'Region'], var_name='Year', value_name='Value')

    # 过滤掉值为0的时间段
    data_filtered = data_filtered[data_filtered['Value'] > 0]

    fig = px.timeline(
        data_filtered,
        x_start="Year",
        x_end="Year",
        y="Country",
        color="Value",
        title=f"Gantt Chart for {feature}",
        labels={'Value': feature, 'Country': 'Country'},
        color_continuous_scale="Viridis"
    )
    st.plotly_chart(fig)

# 主程序调用函数
def main():
    st.title("全球电力统计数据可视化平台")
    uploaded_file = st.file_uploader("Upload your Global Electricity Statistics CSV file", type=["csv"])
    if uploaded_file:
        data = load_data(uploaded_file)

        st.markdown("---")

        # 热力图部分
        st.header("产耗电全球热力图")
        feature_options_heatmap = ["net generation", "net consumption"]
        selected_feature_heatmap = st.selectbox("Select Heatmap Feature", feature_options_heatmap, key="heatmap_feature")
        selected_year_heatmap = st.slider("Select Year", min_value=1980, max_value=2021, value=1980, step=1, key="heatmap_year")
        draw_global_heatmap(data, str(selected_year_heatmap), selected_feature_heatmap)

        st.markdown("---")

        # 折线图部分
        st.header("产耗电折线图")
        draw_combined_line_plots(data)

        st.markdown("---")

        # 气泡图部分
        st.header("产耗电气泡图")
        feature_options_bubble = ["net generation", "net consumption"]
        selected_feature_bubble = st.selectbox("Select Bubble Chart Feature", feature_options_bubble, key="bubble_feature")
        selected_year_bubble = st.slider("Select Year", min_value=1980, max_value=2021, value=1980, step=1, key="bubble_year")
        draw_bubble_chart(data, str(selected_year_bubble), selected_feature_bubble)

        st.markdown("---")

        # 时间序列动画部分
        st.header("产耗电数据时间序列动画")
        feature_options_animation = ["net generation", "net consumption"]
        selected_feature_animation = st.selectbox("Select Feature for Animation", feature_options_animation, key="animation_feature")
        draw_time_series_animation(data, selected_feature_animation)
        st.markdown("---")

        # 动态关系图部分
        # st.header("Network Diagram for Selected Year and Feature")
        st.header("网络关系图")
        selected_feature_network = st.selectbox("Select Network Diagram Feature", feature_options_heatmap,
                                                key="network_feature")
        selected_year_network = st.slider("Select Year", min_value=1980, max_value=2021, value=1980, step=1,
                                          key="network_year")
        draw_network_diagram(data, str(selected_year_network), selected_feature_network)

        st.markdown("---")

        # 多层次环形图部分
        st.header("多层次环形图")
        selected_feature_sunburst = st.selectbox("Select Sunburst Chart Feature", feature_options_heatmap,
                                                 key="sunburst_feature")
        draw_sunburst_chart(data, selected_feature_sunburst)

        st.markdown("---")
        st.header("地理时间动态热力图（Dynamic Map）")
        selected_feature_map = st.selectbox("Select Feature for Dynamic Map", ["net generation", "net consumption"],
                                            key="dynamic_map_feature")
        draw_dynamic_map(data, selected_feature_map)

        st.markdown("---")
        st.header("雷达（Radar Chart）")
        selected_year_radar = st.slider("Select Year for Radar Chart", min_value=1980, max_value=2021, value=1980,
                                        step=1, key="radar_year")
        draw_radar_chart(data, selected_year_radar)

        st.markdown("---")
        st.header("堆叠（Stacked Area Chart）")
        selected_feature_area = st.selectbox("Select Feature for Stacked Area Chart",
                                             ["net generation", "net consumption"], key="area_chart_feature")
        draw_stacked_area_chart(data, selected_feature_area)

        st.markdown("---")
        st.header("桑基（sankey）")
        selected_feature_sankey = st.selectbox("Select Feature for Sankey Diagram",
                                               ["net generation", "net consumption"], key="sankey_feature")
        selected_year_sankey = st.slider("Select Year for Sankey Diagram", min_value=1980, max_value=2021, value=1980,
                                         step=1, key="sankey_year")
        draw_sankey_chart(data, selected_feature_sankey, selected_year_sankey)

        st.markdown("---")
        st.header("热力树图（Treemap）")
        selected_feature_treemap = st.selectbox("Select Feature for Treemap", ["net generation", "net consumption"],
                                                key="treemap_feature")
        selected_year_treemap = st.slider("Select Year for Treemap", min_value=1980, max_value=2021, value=1980, step=1,
                                          key="treemap_year")
        draw_treemap(data, selected_feature_treemap, selected_year_treemap)

        # st.markdown("---")
        # st.header("Parallel Coordinates Plot for Selected Feature")
        # selected_feature_parallel = st.selectbox("Select Feature for Parallel Coordinates Plot",
        #                                          ["net generation", "net consumption"], key="parallel_feature")
        # draw_parallel_coordinates(data, selected_feature_parallel)

        st.markdown("---")
        st.header("径向柱状图 (Radial Bar Chart)")
        selected_feature_radial = st.selectbox("Select Feature for Radial Bar Chart",
                                               ["net generation", "net consumption"], key="radial_feature")
        selected_year_radial = st.slider("Select Year for Radial Bar Chart", min_value=1980, max_value=2021, value=1980,
                                         step=1, key="radial_year")
        draw_radial_bar_chart(data, selected_feature_radial, selected_year_radial)

        # st.markdown("---")
        # st.header("甘特图(Gantt Chart)")
        # selected_feature_gantt = st.selectbox("Select Feature for Gantt Chart", ["net generation", "net consumption"],
        #                                       key="gantt_feature")
        # draw_gantt_chart(data, selected_feature_gantt)


if __name__ == "__main__":
    main()
