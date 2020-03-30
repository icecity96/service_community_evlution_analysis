import numpy as np
import plotly.graph_objects as go


def summary_report(shape_values, feature_names, class_names, merge=False):
    """
    Summary report
    :param class_names:
    :param shape_values:
    :param feature_names:
    :param merge:
    :return:
    """
    if merge:
        importances = []
        for shap_value in shape_values:
            shap_value = shap_value.reshape((shap_value.shape[0], 3, -1))
            importance = np.sum(shap_value, axis=1)
            importance = np.sum(np.abs(importance), axis=0)
            importances.append(importance.tolist())
        fig = go.Figure(data=go.Heatmap(
            z = importances,
            x = feature_names,
            y = class_names
        ))
        fig.write_html("figure/summary_merge.html")
    else:
        importances = []
        for shap_value in shape_values:
            importance = np.sum(np.abs(shap_value), axis=0)
            importances.append(importance.tolist())
        fig = go.Figure(data=go.Heatmap(
            z = importances,
            x = feature_names,
            y = class_names
        ))
        fig.write_html("figure/summary_without_merge.html")


def dependency_report(feature_name, class_name, shap_values, data, feature_names, class_names, relative=True):
    feature_index, class_index = feature_names.index(feature_name), class_names.index(class_name)
    shap_value = shap_values[class_index]
    shap_value = [sum([value[feature_index], value[feature_index + len(feature_names)],
                    value[feature_index + 2 * len(feature_names)]]) for value in shap_value]
    initial_value = [d[feature_index] for d in data]
    second_value = [d[feature_index + len(feature_names)] - d[feature_index] for d in data]
    third_value = [d[feature_index + 2 * len(feature_names)] - d[feature_index + len(feature_names)] for d in data]
    symbols = ["star-triangle-up" if d[feature_index + 2*len(feature_names)] - d[feature_index + len(feature_names)] > 0 else "star-triangle-down" for d in data]
    fig = go.Figure(data=[go.Scatter(
        x = initial_value,
        y = shap_value,
        marker_symbol=symbols,
        mode='markers',
        marker= dict(
            size=[abs(v) for v in third_value],
            color=second_value,
            colorscale='Viridis',
            colorbar=dict(
                title=f"2-{feature_name} - 1-{feature_name}",
            ),
            sizemode='area',
            sizeref=2. * max(second_value) / (20. ** 2),
            sizemin=4,
            showscale=True
        )
    )])
    fig.update_layout(
        xaxis=dict(
            title=f"1-{feature_name} value",
            gridcolor='white',
            gridwidth=2,
        ),
        yaxis=dict(
            title=f"shap value for {feature_name} (sum)",
            gridcolor='white',
            gridwidth=2,
        ),

    )
    fig.write_html(f"figure/dependency_{feature_name}_{class_name}.html")