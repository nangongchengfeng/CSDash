# -*- coding: utf-8 -*-
# @Time    : 2023/5/10 14:34
# @Author  : 南宫乘风
# @Email   : 1794748404@qq.com
# @File    : app.py
# @Software: PyCharm
from csdn_spider import get_info, get_blog
from dash import dcc
import dash
from dash import html
import pandas as pd
import plotly.graph_objs as go
from dash.dependencies import Input, Output
import datetime as dt
from sqlalchemy import create_engine
from flask_caching import Cache

# 今天的时间
today = dt.datetime.today().strftime("%Y-%m-%d")

# 连接数据库
engine = create_engine('mysql+pymysql://root:123456@192.168.102.20/csdn?charset=utf8')


# 创建一个实例
app = dash.Dash(__name__, external_stylesheets=['../static/css/my.css', '../static/css/skeleton.min.css',
                                                '../static/css/bootstrap.min.css'])
server = app.server

# 可以选择使用缓存, 减少频繁的数据请求
cache = Cache(app.server, config={
    'CACHE_TYPE': 'filesystem',
    'CACHE_DIR': 'cache-directory'
})

# 读取info表的数据
info = pd.read_sql('info', con=engine)

# print(info)
# 图表颜色
color_scale = ['#2c0772', '#3d208e', '#8D7DFF', '#CDCCFF', '#C7FFFB', '#ff2c6d', '#564b43', '#161d33']


def indicator(text, id_value):
    """第一列的文字及数字信息显示"""
    return html.Div([
        html.P(text, className="twelve columns indicator_text"),
        html.P(id=id_value, className="indicator_value"),
    ], className="col indicator")


def get_news_table(data):
    """获取文章列表展示"""
    if data is None or len(data) == 0:
        return html.Table([
            html.Tbody([
                html.Tr([
                    html.Td("暂无数据")
                ])
            ])
        ], style={"height": "90%", "width": "98%"})

    try:
        # 创建数据副本
        df = data.copy()

        # 确保日期格式正确
        try:
            df['date'] = pd.to_datetime(df['date'])
            # 格式化日期
            dates = df['date'].dt.strftime('%Y-%m-%d %H:%M:%S').tolist()
            show_dates = True
        except:
            print("Warning: Failed to convert dates")
            show_dates = False

        titles = df['title'].tolist()
        urls = df['url'].tolist()

        # 根据是否成功转换日期来决定显示格式
        if show_dates:
            return html.Table([html.Tbody([
                html.Tr([
                    html.Td([
                        html.A(titles[i], href=urls[i], target="_blank"),
                        html.Span(
                            dates[i],
                            style={'marginLeft': '10px', 'color': '#666', 'fontSize': '12px'}
                        )
                    ])
                ], style={'height': '30px', 'fontSize': '16'}) for i in range(min(len(df), 100))
            ])], style={"height": "90%", "width": "98%"})
        else:
            # 如果日期转换失败，只显示标题和链接
            return html.Table([html.Tbody([
                html.Tr([
                    html.Td(
                        html.A(titles[i], href=urls[i], target="_blank")
                    )
                ], style={'height': '30px', 'fontSize': '16'}) for i in range(min(len(df), 100))
            ])], style={"height": "90%", "width": "98%"})

    except Exception as e:
        print(f"Error in get_news_table: {str(e)}")
        # 发生错误时返回错误提示
        return html.Table([
            html.Tbody([
                html.Tr([
                    html.Td("数据显示出错")
                ])
            ])
        ], style={"height": "90%", "width": "98%"})


@cache.memoize(timeout=3590)
def get_catego():
    """获取当日最新的文章数据"""
    df = pd.read_sql("categorize", con=engine)
    return df


@cache.memoize(timeout=3590)
def get_df():
    """获取当日最新的文章数据"""
    try:
        df = pd.read_sql(today, con=engine)

        # 转换日期格式
        df['date'] = pd.to_datetime(df['date'])
        df['date_day'] = pd.to_datetime(df['date'].dt.date)
        df['date_month'] = df['date'].dt.strftime('%Y年%m月')
        df['weekday'] = df['date_day'].dt.weekday
        df['year'] = df['date_day'].dt.year
        df['month'] = df['date_day'].dt.month
        df['week'] = df['date_day'].dt.isocalendar().week

        return df
    except Exception as e:
        print(f"Error in get_df: {str(e)}")
        return pd.DataFrame()


# 导航栏的图片及标题
head = html.Div(
    [
        html.Div(
            html.A(
                html.Img(
                    src='https://www.ownit.top/img/avatar_hu227367ba8544f2fc7811ed9508937bec_102665_300x0_resize_box_3.png',
                    style={"width": "100%", "height": "100%", "object-fit": "contain", "border-radius": "50%"}),
                href="https://blog.csdn.net/heian_99"
            ),
            style={"float": "left", "height": "90%", "margin-top": "5px", "margin-right": "10px"}
        ),
        html.A(html.Span("{}博客的Dashboard".format(info['author_name'][0]), className='app-title'),
               href="https://blog.csdn.net/heian_99",
               style={'display': 'flex', 'justify-content': 'center', 'align-items': 'center'}),

    ],
    className="row header"
)

# 第一列的文字及数字信息
columns = info.columns[3:]
col_name = ['文章数', '关注数', '喜欢数', '评论数', '等级', '访问数', '积分', '排名']
row1 = html.Div([
    indicator(col_name[i], col) for i, col in enumerate(columns)
], className='row')

# 第二列
row2 = html.Div([
    html.Div([
        html.P("每月文章写作情况"),
        dcc.Graph(id="bar", style={"height": "90%", "width": "98%"}, config=dict(displayModeBar=False), )
    ], className="col-4 chart_div", ),
    html.Div([
        html.P("各类型文章占比情况"),
        dcc.Graph(id="pie", style={"height": "90%", "width": "98%"}, config=dict(displayModeBar=False), )
    ], className="col-4 chart_div"),
    html.Div([
        html.P("各类型文章阅读情况"),
        dcc.Graph(id="mix", style={"height": "90%", "width": "98%"}, config=dict(displayModeBar=False), )
    ], className="col-4 chart_div", )
], className='row')

# 年数统计, 我的是2019 2020 2021
years = get_df()['year'].unique()
select_list = ['每月文章', '类型占比', '类型阅读量', '每日情况']
latest_year = max(years) if len(years) > 0 else dt.datetime.now().year
# 两个可交互的下拉选项
# 两个可交互的下拉选项
dropDowm1 = html.Div([
    html.Div([
        dcc.Dropdown(id='dropdown1',
                    options=[{'label': '{}年'.format(year), 'value': year} for year in sorted(years, reverse=True)],  # 降序排列年份
                    value=latest_year,  # 使用最新年份作为默认值
                    style={'width': '40%'})
    ], className='col-6', style={'padding': '2px', 'margin': '0px 5px 0px'}),
    html.Div([
        dcc.Dropdown(id='dropdown2',
                    options=[{'label': select_list[i], 'value': item} for i, item in
                            enumerate(['bar', 'pie', 'mix', 'heatmap'])],
                    value='heatmap', style={'width': '40%'})
    ], className='col-6', style={'padding': '2px', 'margin': '0px 5px 0px'})
], className='row')

# 第三列
row3 = html.Div([
    html.Div([
        html.P("每日写作情况"),
        dcc.Graph(id="heatmap", style={"height": "90%", "width": "98%"}, config=dict(displayModeBar=False), )
    ], className="col-6 chart_div", ),
    html.Div([
        html.P("文章列表"),
        html.Div(
            get_news_table(
                get_df()
                .assign(date=lambda x: pd.to_datetime(x['date']))
                .sort_values('date', ascending=False)
            ),
            id='click-data'
        ),
    ], className="col-6 chart_div", style={"overflowY": "scroll"})
], className='row')

# 总体情况
app.layout = html.Div([
    # 定时器
    dcc.Interval(id="stream", interval=1000 * 60, n_intervals=0),
    dcc.Interval(id="river", interval=1000 * 60 * 60, n_intervals=0),
    html.Div(id="load_info", style={"display": "none"}, ),
    html.Div(id="load_click_data", style={"display": "none"}, ),
    head,
    html.Div([
        row1,
        row2,
        dropDowm1,
        row3,
    ], style={'margin': '0% 30px'}),
])


# 回调函数, 60秒刷新info数据, 即第一列的数值实时刷新
@app.callback(Output('load_info', 'children'), [Input("stream", "n_intervals")])
def load_info(n):
    try:
        df = pd.read_sql('info', con=engine)
        return df.to_json()
    except:
        pass


# 回调函数, 60分钟刷新今日数据, 即第二、三列的数值实时刷新(爬取文章数据, 并写入数据库中)
@app.callback(Output('load_click_data', 'children'), [Input("river", "n_intervals")])
def cwarl_data(n):
    if n != 0:
        df_article = get_blog()
        df_article.to_sql(today, con=engine, if_exists='replace', index=True)


# 回调函数, 第一个柱状图
@app.callback(Output('bar', 'figure'), [Input("river", "n_intervals")])
def get_bar(n):
    df = get_df()
    df_date_month = pd.DataFrame(df['date_month'].value_counts(sort=False))
    df_date_month.sort_index(inplace=True)
    # print(df_date_month)
    date_month_list = df_date_month.index.tolist()  # 将date_month列转换为列表
    count_list = df_date_month['count'].tolist()  # 将count列转换为列表

    trace = go.Bar(
        x=date_month_list,
        y=count_list,
        text=count_list,
        textposition='auto',
        marker=dict(color=color_scale[:len(date_month_list)])
    )
    layout = go.Layout(
        margin=dict(l=40, r=40, t=10, b=50)
    )
    return go.Figure(data=[trace], layout=layout)


# 回调函数, 中间的饼图
@app.callback(Output('pie', 'figure'), [Input("river", "n_intervals")])
def get_pie(n):
    df = get_df()
    df_types = pd.DataFrame(df.groupby('type')['type'].count())
    df_types.columns = ['count']

    trace = go.Pie(
        labels=df_types.index,
        values=df_types['count'],
        marker=dict(colors=color_scale[:len(df_types.index)])
    )
    layout = go.Layout(
        margin=dict(l=50, r=50, t=50, b=50),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
    )
    return go.Figure(data=[trace], layout=layout)


# 回调函数, 左下角热力图
@app.callback(Output('heatmap', 'figure'),
              [Input("dropdown1", "value"), Input('river', 'n_intervals')])
def get_heatmap(value, n):
    df = get_df()
    grouped_by_year = df.groupby('year')
    data = grouped_by_year.get_group(value)
    cross = pd.crosstab(data['weekday'], data['week'])
    cross.sort_index(inplace=True)
    trace = go.Heatmap(
        x=['第{}周'.format(i) for i in cross.columns],
        y=["星期{}".format(i + 1) if i != 6 else "星期日" for i in cross.index],
        z=cross.values,
        colorscale="Blues",
        reversescale=False,
        xgap=4,
        ygap=5,
        showscale=False
    )
    layout = go.Layout(
        margin=dict(l=50, r=40, t=30, b=50),
    )
    return go.Figure(data=[trace], layout=layout)


# 回调函数, 第二个柱状图(柱状图+折线图)
@app.callback(Output('mix', 'figure'), [Input("river", "n_intervals")])
def get_mix(n):
    df = get_catego()
    print("测试mix", df)
    df_type_visit_sum = pd.DataFrame(df['read_num'].groupby(df['categorize']).sum())
    # df_type_visit_sum = pd.DataFrame(df[['read_num','categorize']])
    df_type_visit_sum = df_type_visit_sum.sort_values(by='read_num', ascending=False).nlargest(15, 'read_num')

    trace1 = go.Bar(
        x=df_type_visit_sum.index,
        y=df_type_visit_sum['read_num'],
        name='总阅读',
        marker=dict(color='#ffc97b'),
        yaxis='y',
    )
    trace2 = go.Scatter(
        x=df_type_visit_sum.index,
        y=df_type_visit_sum.index,
        name='平均阅读',
        yaxis='y2',
        line=dict(color='#161D33')
    )
    layout = go.Layout(
        margin=dict(l=60, r=60, t=30, b=50),
        showlegend=False,
        yaxis=dict(
            side='left',
            title='阅读总数',
            gridcolor='#e2e2e2'
        ),
        yaxis2=dict(
            showgrid=False,  # 网格
            title='阅读平均',
            anchor='x',
            overlaying='y',
            side='right'
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
    )
    return go.Figure(data=[trace1], layout=layout)


# 点击事件, 选择两个下拉选项, 点击对应区域的图表, 文章列表会刷新
@app.callback(Output('click-data', 'children'),
              [Input('pie', 'clickData'),
               Input('bar', 'clickData'),
               Input('mix', 'clickData'),
               Input('heatmap', 'clickData'),
               Input('dropdown1', 'value'),
               Input('dropdown2', 'value')])
def display_click_data(pie, bar, mix, heatmap, d_value, fig_type):
    try:
        df = get_df()
        data = None

        # 根据点击事件进行过滤
        if fig_type == 'pie' and pie is not None:
            type_value = pie['points'][0]['label']
            data = df[df['type'] == type_value]

        elif fig_type == 'bar' and bar is not None:
            date_month_value = bar['points'][0]['x']
            data = df[df['date_month'] == date_month_value]

        elif fig_type == 'mix' and mix is not None:
            type_value = mix['points'][0]['x']
            data = df[df['type'] == type_value]

        elif fig_type == 'heatmap' and heatmap is not None:
            z = heatmap['points'][0]['z']
            if z > 0:
                week = heatmap['points'][0]['x'][1:-1]
                weekday = heatmap['points'][0]['y'][-1]
                if weekday == '日':
                    weekday = 6
                else:
                    weekday = int(weekday) - 1
                year = d_value
                data = df[(df['weekday'] == weekday) &
                          (df['week'] == int(week)) &
                          (df['year'] == year)]

        # 如果没有点击事件，显示所有文章（按时间排序）
        if data is None:
            data = df

        # 对数据进行处理
        data = data.copy()
        # URL去重
        data = data.drop_duplicates(subset=['url'], keep='first')
        # 转换日期格式
        data['date'] = pd.to_datetime(data['date'])
        # 按时间排序
        data = data.sort_values('date', ascending=False)

        return get_news_table(data)

    except Exception as e:
        print(f"Error in display_click_data: {str(e)}")
        # 发生错误时返回按时间排序的所有文章
        return get_news_table(
            get_df()
            .drop_duplicates(subset=['url'], keep='first')
            .sort_values('date', ascending=False)
        )


# 第一列的数值
def update_info(col):
    def get_data(json, n):
        df = pd.read_json(json)
        return df[col][0]

    return get_data


for col in columns:
    app.callback(Output(col, "children"),
                 [Input('load_info', 'children'), Input("stream", "n_intervals")]
                 )(update_info(col))

if __name__ == '__main__':
    # debug模式, 端口7777
    # app.run_server(debug=True, threaded=True, port=7777)
    # 正常模式, 网页右下角的调试按钮将不会出现
    print("start server")
    app.run(host="0.0.0.0", port=7777)
