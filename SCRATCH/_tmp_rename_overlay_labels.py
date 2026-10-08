from pathlib import Path

p = Path(r'h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\General\Pipeline\Stages\DisplayFunctions\DecoderPredictionError.py')
text = p.read_text(encoding='utf-8')
replacements = [
    ('WeightedCorrelationPaginatedPlotDataProvider', 'OverlayLabelsPaginatedPlotDataProvider'),
    ('WeightedCorrelationPlotData', 'OverlayLabelsPlotData'),
    ('decoder_build_single_weighted_correlation_data', 'decoder_build_single_overlay_labels_data'),
    ('enable_weighted_correlation_info', 'enable_overlay_labels_info'),
    ('enable_weighted_corr_data_provider_modify_axes_rect', 'enable_overlay_labels_modify_axes_rect'),
    ('weighted_corr_text_original_figure_rect', 'overlay_labels_text_original_figure_rect'),
    ('weighted_corr_text_was_axes_rect_modified', 'overlay_labels_text_was_axes_rect_modified'),
    ('plots_data.weighted_corr_data', 'plots_data.overlay_labels_data'),
    ("{'weighted_corr_data': None}", "{'overlay_labels_data': None}"),
    ("{'weighted_corr': {}}", "{'overlay_labels': {}}"),
    ("plots_group_identifier_key: str = 'weighted_corr'", "plots_group_identifier_key: str = 'overlay_labels'"),
    ("{'plot_wcorr_data':", "{'plot_overlay_labels_data':"),
    ("plots['weighted_corr']", "plots['overlay_labels']"),
]
for old, new in replacements:
    c = text.count(old)
    text = text.replace(old, new)
    print(f'{old!r} -> {c} replacements')
p.write_text(text, encoding='utf-8')
print('done')
