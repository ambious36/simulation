# app.py
import numpy as np
import panel as pn
from bokeh.plotting import figure
from bokeh.models import ColorBar
from bokeh.palettes import Viridis256
from bokeh.transform import linear_cmap
from bokeh.layouts import gridplot
import param

# 导入您提供的 simulation 模块中的函数
# 注意：需确保 simulation.py 在当前目录
from simulation import run_simulation

pn.extension(sizing_mode="stretch_width")

# ----------------------------
# 参数类：用于控制输入
# ----------------------------
class SimulationController(param.Parameterized):
    
    seed = param.Integer(default=2025, bounds=(1, 10000), label="Policy Seed")
    T_max = param.Integer(default=100, bounds=(10, 200), label="Simulation Periods(<=200)")
    n_regions = param.Integer(default=10, bounds=(10, 500), label="Number of Regions (n_region)")
    max_capacity = param.Number(default=200, bounds=(1, 1000), step=100, label="Max Production Capacity")
    pollution_intensity = param.Number(default=0.035, bounds=(0.0, 2.0), step=0.05, label="Pollution Intensity")
    abatement_efficiency = param.Number(default=0.45, bounds=(0.0, 1.0), step=0.05, label="Abatement Efficiency")
    base_repair_fraction = param.Number(default=0.03, bounds=(0.0, 0.5), step=0.01, label="Base Repair Fraction")
    green_tech_reduction = param.Number(default=0.45, bounds=(0.0, 1.0), step=0.05, label="Green Tech Reduction")
    min_activation_periods = param.Number(default=10, bounds=(1,100), step=1,label="min_activation_periods")
    
    run = param.Action(lambda x: x.param.trigger('run'), label="Run Simulation")

    # 存储结果
    #results = param.Dict(default=None)
    #subsidy_res = param.Dict(default=None)
    results = param.Tuple(default=(None, None))

    '''def __init__(self, **params):
        super().__init__(**params)
        self.results = None
        self.subsidy_res = None'''

    
    def _run_one(self, policy, seed,T_max,n_regions,
            max_capacity,pollution_intensity,abatement_efficiency,
            base_repair_fraction,green_tech_reduction):
        return run_simulation(policy_mode=policy, seed=seed,
            T_max=T_max, n_regions=n_regions,
            max_capacity=max_capacity, pollution_intensity=pollution_intensity,
            abatement_efficiency=abatement_efficiency,
            base_repair_fraction=base_repair_fraction,
            green_tech_reduction=green_tech_reduction)
    
    @param.depends('run', watch=True)    
    def _run_simulations(self):
        self.carbon_res = self._run_one(
            policy="carbon_tax",
            seed=self.seed,
            T_max=self.T_max,
            n_regions=self.n_regions,
            max_capacity=self.max_capacity,
            pollution_intensity=self.pollution_intensity,
            abatement_efficiency=self.abatement_efficiency,
            base_repair_fraction=self.base_repair_fraction,
            green_tech_reduction=self.green_tech_reduction
        )
        self.subsidy_res = self._run_one(
            policy="green_subsidy",
            seed=self.seed,
            T_max=self.T_max,
            n_regions=self.n_regions,
            max_capacity=self.max_capacity,
            pollution_intensity=self.pollution_intensity,
            abatement_efficiency=self.abatement_efficiency,
            base_repair_fraction=self.base_repair_fraction,
            green_tech_reduction=self.green_tech_reduction
        )
        self.results = (self.carbon_res, self.subsidy_res)

# ----------------------------
# 绘图函数（Bokeh 版）
# ----------------------------
def make_plot(title, x, y1, y2, xlabel="Period", ylabel="", colors=("steelblue", "orange")):
    p = figure(title=title, height=250, tools="hover,pan,wheel_zoom,reset")
    p.line(x, y1, legend_label="Carbon Tax", color=colors[0], line_width=2)
    p.line(x, y2, legend_label="Green Subsidy", color=colors[1], line_width=2)
    p.xaxis.axis_label = xlabel
    p.yaxis.axis_label = ylabel
    p.legend.location = "top_left"
    p.grid.visible = True
    return p

def make_scatter(geo_pos, final_y, title):
    mapper = linear_cmap(field_name='output', palette=Viridis256, low=min(final_y), high=max(final_y))
    p = figure(title=title, height=250, tools="hover,pan,wheel_zoom,reset")
    scatter = p.scatter(
        x=geo_pos[:, 0],
        y=geo_pos[:, 1],
        size=12,
        color=mapper,
        line_color="black",
        alpha=0.8
    )
    color_bar = ColorBar(color_mapper=mapper['transform'], width=8, location=(0,0))
    p.add_layout(color_bar, 'right')
    p.hover.tooltips = [("Output", "@output{0.2f}")]
    # Add output as column data source for hover
    from bokeh.models import ColumnDataSource
    source = ColumnDataSource(data=dict(x=geo_pos[:,0], y=geo_pos[:,1], output=final_y))
    scatter.data_source = source
    return p

def make_histogram(final_y1, final_y2, title):
    hist1, edges1 = np.histogram(final_y1, bins=6, density=False)
    hist2, edges2 = np.histogram(final_y2, bins=6, density=False)
    p = figure(title=title, height=250)
    p.quad(top=hist1, bottom=0, left=edges1[:-1], right=edges1[1:], 
           fill_color="steelblue", line_color="white", alpha=0.7, legend_label="Carbon Tax")
    p.quad(top=hist2, bottom=0, left=edges2[:-1], right=edges2[1:], 
           fill_color="orange", line_color="white", alpha=0.7, legend_label="Green Subsidy")
    p.xaxis.axis_label = "Output"
    p.yaxis.axis_label = "Frequency"
    p.legend.location = "top_right"
    p.grid.visible = True
    return p

def make_activation_plot(subsidy_active, title):
    if not subsidy_active:
        p = figure(title=title, height=250)
        p.text([0.5], [0.5], ["No data"], text_align="center", text_baseline="middle")
        return p
    x = list(range(len(subsidy_active)))
    y = subsidy_active
    p = figure(title=title, height=250, y_range=(-0.1, 1.1))
    p.circle(x, y, size=6, color="green")
    p.line(x, y, color="green", alpha=0.6)
    p.yaxis.ticker = [0, 1]
    p.grid.visible = True
    return p

def make_score_plot(scores, activation, title):
    if not scores:
        p = figure(title=title, height=250)
        p.text([0.5], [0.5], ["No score data"], text_align="center", text_baseline="middle")
        return p
    x = list(range(len(scores)))
    p = figure(title=title, height=250)
    p.line(x, scores, color="blue", legend_label="Composite Score")
    p.line(x, [0.2]*len(x), color="red", line_dash="dashed", legend_label="Threshold")
    # Highlight active periods
    active_x = [i for i, a in enumerate(activation) if a]
    active_y = [scores[i] for i in active_x]
    if active_x:
        p.circle(active_x, active_y, size=6, color="green", legend_label="Activated")
    p.legend.location = "top_left"
    p.grid.visible = True
    return p

# ----------------------------
# 主 Panel 布局
# ----------------------------
controller = SimulationController()

# 占位符（初始空图）
empty_fig = figure(height=250)
empty_fig.text([0.5], [0.5], ["Run simulation to generate results"], text_align="center", text_baseline="middle")

@pn.depends(controller.param.results)
def update_plots(results):
    carbon_res, subsidy_res = results
    if carbon_res is None or subsidy_res is None:
        plots = [empty_fig] * 9
    else:
        T = len(carbon_res['total_output'])
        x = list(range(T))

        p1 = make_plot("Total Output", x, carbon_res['total_output'], subsidy_res['total_output'])
        p2 = make_plot("Environmental Capital", x, carbon_res['env_levels'], subsidy_res['env_levels'], ylabel="E")
        p3 = make_plot("Gini Coefficient", x, carbon_res['gini_list'], subsidy_res['gini_list'], ylabel="Gini")
        p4 = make_plot("Cooperation Clusters", x, carbon_res['cluster_counts'], subsidy_res['cluster_counts'])

        p5 = make_scatter(carbon_res['geo_pos'], carbon_res['final_y'], "Final Output Map (Carbon Tax)")
        p6 = make_scatter(subsidy_res['geo_pos'], subsidy_res['final_y'], "Final Output Map (Green Subsidy)")

        p7 = make_histogram(carbon_res['final_y'], subsidy_res['final_y'], "Regional Output Distribution")

        p8 = make_activation_plot(subsidy_res.get('subsidy_active', []), "Green Subsidy Activation")

        p9 = make_score_plot(
            subsidy_res.get('composite_score_history', []),
            subsidy_res.get('subsidy_active', []),
            "Green Subsidy Composite Score"
        )

        plots = [p1, p2, p3, p4, p5, p6, p7, p8, p9]

    # 3x3 grid
    grid = gridplot([[plots[0], plots[1], plots[2]],
                     [plots[3], plots[4], plots[5]],
                     [plots[6], plots[7], plots[8]]],
                    toolbar_location="right")
    return grid

# 构建界面
input_panel = pn.Param(
    controller,
    parameters=['seed', 'T_max',
                'n_regions', 'max_capacity', 'pollution_intensity',
                'abatement_efficiency', 'base_repair_fraction', 'green_tech_reduction', 
                'run'],
    widgets={
        'run': {'button_type': 'success'}
    }
)

main_area = pn.panel(update_plots, loading_indicator=True)

app = pn.Column(
    pn.Row(
        pn.Column("## 🎛️ Simulation Parameters", input_panel, width=320),
        pn.Spacer(width=20),
        pn.Column("## 📊 Results Dashboard", main_area),
    ),
    sizing_mode="stretch_both"
)

# 启动服务器（或直接在 notebook 中显示）
if __name__ == "__main__":
    app.show()
else:
    app.servable()