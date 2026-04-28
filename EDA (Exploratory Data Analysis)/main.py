
# <editor-fold desc="Import Section">
from config import *
from use_pickles import load_pickle
import numpy as np
import os

# 2.1) Dataset Overview
from data_overview import (
    inspect_data_format,
    inspect_labels_format,
    get_dataset_scale,
    print_data_format_report,
    print_dataset_scale_report,
    export_dataset_scale_to_csv,
)

# 2.2) Raw analysis
from raw_signal_analysis import (
    analyze_signal_composition,
    print_signal_composition,
    export_signal_composition,
    analyze_temporal_structure,
    analyze_label_intervals,
    check_signal_continuity,
    plot_example_signal,
    plot_signal_with_labels_single,
    plot_signal_with_labels_all,
    print_temporal_structure_report,
    print_interval_summary_by_activity,
)

# 2.2.3 Signal Behaviour
from signal_behaviour import (
    run_signal_behaviour_section,
    get_group_segments,
    plot_activity_examples,
    plot_activity_overlay,
    plot_group_representatives,
    plot_comparison_panel,
    plot_activity_groups_overlay
)

# 2.2.4 Signal Stats
from statistical_properties import (
    build_segment_feature_table,
    print_segment_feature_overview,
    summarize_by_group,
    summarize_by_activity,
    print_group_summary,
    print_activity_summary,
    export_segment_feature_outputs,
    summarize_by_subject,
    build_subject_channel_summary,
    print_subject_summary,
    export_subject_outputs,
    compute_activity_similarity,
    summarize_candidate_outliers,
    print_similarity_summary,
    print_top_candidate_outliers,
    export_similarity_outputs,
)

#2.2.5
from per_subject_signals import (
    summarize_subject_contribution,
    build_per_subject_channel_summary,
    print_subject_contribution_summary,
    export_per_subject_outputs,
    plot_subject_variability_bars,
    plot_same_activity_across_subjects,
    plot_same_activity_across_subjects_gyro,
    plot_per_subject_multibar,
)

#2.2.6
from channel_correlation import (
    compute_channel_correlation,
    plot_channel_correlation_heatmap,
    plot_grouped_channel_correlation,
    print_channel_correlation_summary,
)

#2.3
from outlier_investigation import (
    extract_activity_segments_with_metadata,
    plot_activity_outliers_vs_typicals_acc,
    plot_activity_outliers_vs_typicals_gyro,
    plot_top_global_outliers,
)


from data_cache import (
    load_raw_recordings,
    build_segment_cache_from_manifest,
    save_cache_bundle,
    make_segment_id,
)

# </editor-fold>

# <editor-fold desc="Conditional Execution Section">
#2.1
Perform_DatasetOverview = 0

#2.2
Perform_RawAnalysis = 0
Plot_all_signals_with_labels = 0

#2.2.3
Perform_SignalBehaviour = 0

#2.2.4
Perform_StatisticalAnalysis = 0

#2.2.5
Perform_PerSubjectAnalysis = 0

#2.2.6
Perform_ChannelCorrelation = 1

# 2.3
Perform_OutlierInvestigation = 0
#2.2.4 must run with 2.3 - should add the similarity df in order to make it run seperately


# </editor-fold>





############################################
# Execution
############################################

# 2.1
if Perform_DatasetOverview == 1:
    # -------------------------
    # 2.1.3 Data Format
    # -------------------------
    format_info = inspect_data_format(
        dataset_dir=DatasetDir,
        acc_pattern=ACC_PATTERN,
        gyro_pattern=GYRO_PATTERN,
    )

    labels_info = inspect_labels_format(
        dataset_dir=DatasetDir,
        labels_file=LABELS_FILE,
    )

    print_data_format_report(format_info, labels_info)

    # -------------------------
    # 2.1.4 Dataset Scale
    # -------------------------
    scale_info = get_dataset_scale(
        dataset_dir=DatasetDir,
        acc_pattern=ACC_PATTERN,
        gyro_pattern=GYRO_PATTERN,
        sampling_rate=SAMPLING_RATE,
    )

    print_dataset_scale_report(
        scale_info,
        print_per_subject=PRINT_PER_SUBJECT,
        print_per_experiment=PRINT_PER_EXPERIMENT,
        print_per_file=PRINT_PER_FILE,
    )

    export_dataset_scale_to_csv(
        scale_info=scale_info,
        output_dir=DatasetOverview_OutputDir_2_1,
        sampling_rate=SAMPLING_RATE,
        export_per_file=EXPORT_PER_FILE_CSV,
    )

#2.2
if Perform_RawAnalysis == 1:
    stats = analyze_signal_composition(
        dataset_dir=DatasetDir,
        acc_pattern=ACC_PATTERN,
        gyro_pattern=GYRO_PATTERN,
    )

    print_signal_composition(stats)
    # export_signal_composition(stats, DatasetOverview_OutputDir_2_1)

    recording_info = analyze_temporal_structure(
        dataset_dir=DatasetDir,
        acc_pattern=ACC_PATTERN,
        gyro_pattern=GYRO_PATTERN,
        sampling_rate=SAMPLING_RATE,
    )

    intervals = analyze_label_intervals(
        dataset_dir=DatasetDir,
        labels_file=LABELS_FILE,
        sampling_rate=SAMPLING_RATE,
    )

    continuity_report = check_signal_continuity(
        dataset_dir=DatasetDir,
        acc_pattern=ACC_PATTERN,
    )

    print_temporal_structure_report(
        recording_info=recording_info,
        intervals=intervals,
        continuity_report=continuity_report,
        sampling_rate=SAMPLING_RATE,
    )
    print_interval_summary_by_activity(intervals)

    # plot_example_signal(
    #     dataset_dir=DatasetDir, exp=1, user=1)

    plot_signal_with_labels_single(
        dataset_dir=DatasetDir,
        labels_file=LABELS_FILE,
        exp=1,
        user=1,
    )

    if Plot_all_signals_with_labels == 1:
        plot_signal_with_labels_all(
            dataset_dir=DatasetDir,
            labels_file=LABELS_FILE,
            padding_constant=0.8,
        )

#2.2.3
if Perform_SignalBehaviour == 1:
    # run_signal_behaviour_section(
    #     dataset_dir=DatasetDir,
    #     labels_file=LABELS_FILE,
    #     acc_pattern=ACC_PATTERN,
    #     gyro_pattern=GYRO_PATTERN,
    # )
    #
    # # -------------------------
    # # Static activities
    # # -------------------------
    # static_segments = get_group_segments(
    #     dataset_dir=DatasetDir,
    #     labels_file=LABELS_FILE,
    #     acc_pattern=ACC_PATTERN,
    #     gyro_pattern=GYRO_PATTERN,
    #     activity_ids=[4, 5, 6],
    # )
    #
    # plot_activity_examples(
    #     group_segments=static_segments,
    #     group_name="Static",
    #     num_examples=3,
    #     channels=(0, 3),   # acc_x, gyro_x
    # )
    #
    # # -------------------------
    # # Dynamic activities
    # # -------------------------
    # dynamic_segments = get_group_segments(
    #     dataset_dir=DatasetDir,
    #     labels_file=LABELS_FILE,
    #     acc_pattern=ACC_PATTERN,
    #     gyro_pattern=GYRO_PATTERN,
    #     activity_ids=[1, 2, 3],
    # )
    #
    # plot_activity_examples(
    #     group_segments=dynamic_segments,
    #     group_name="Dynamic",
    #     num_examples=3,
    #     channels=(0, 3),
    # )
    #
    # # -------------------------
    # # Transition activities
    # # -------------------------
    # transition_segments = get_group_segments(
    #     dataset_dir=DatasetDir,
    #     labels_file=LABELS_FILE,
    #     acc_pattern=ACC_PATTERN,
    #     gyro_pattern=GYRO_PATTERN,
    #     activity_ids=[7, 8, 9, 10, 11, 12],
    # )
    #
    # plot_activity_examples(
    #     group_segments=transition_segments,
    #     group_name="Transition",
    #     num_examples=3,
    #     channels=(0, 3),
    # )
    #
    # # -------------------------
    # # Comparison plots
    # # -------------------------
    # plot_group_representatives(
    #     dataset_dir=DatasetDir,
    #     labels_file=LABELS_FILE,
    #     acc_pattern=ACC_PATTERN,
    #     gyro_pattern=GYRO_PATTERN,
    #     channels=(0, 3),
    # )
    #
    # plot_comparison_panel(
    #     dataset_dir=DatasetDir,
    #     labels_file=LABELS_FILE,
    #     acc_pattern=ACC_PATTERN,
    #     gyro_pattern=GYRO_PATTERN,
    #     channel_index=0,   # acc_x
    # )
    plot_activity_groups_overlay(
        dataset_dir=DatasetDir,
        groups=activity_groups,
        num_examples=5,
        channel_index=3,
    )


    # Optional overlays for within-class consistency
    # Example: standing, walking, sit-to-stand
    # for activity_id in [1,2,3,4,5,6,7,8,9,10,11,12]:
    #     activity_segments = get_group_segments(
    #         dataset_dir=DatasetDir,
    #         labels_file=LABELS_FILE,
    #         acc_pattern=ACC_PATTERN,
    #         gyro_pattern=GYRO_PATTERN,
    #         activity_ids=[activity_id],
    #     )[activity_id]
    #
    #     plot_activity_overlay(
    #         segments=activity_segments,
    #         activity_name=label_to_class[activity_id],
    #         num_examples=5,
    #         channel_index=0,
    #     )

#2.2.4
if Perform_StatisticalAnalysis == 1:
    segment_df = build_segment_feature_table(
        dataset_dir=DatasetDir,
        labels_file=LABELS_FILE,
        acc_pattern=ACC_PATTERN,
        gyro_pattern=GYRO_PATTERN,
    )

    print_segment_feature_overview(segment_df)

    group_summary_df = summarize_by_group(segment_df)
    print_group_summary(group_summary_df)

    activity_summary_df = summarize_by_activity(segment_df)
    print_activity_summary(activity_summary_df)

    export_segment_feature_outputs(
        segment_df=segment_df,
        group_summary_df=group_summary_df,
        activity_summary_df=activity_summary_df,
        output_dir=DatasetOverview_OutputDir_2_1,
    )

    # -------------------------
    # Subject statistics
    # -------------------------
    subject_summary_df = summarize_by_subject(segment_df)
    print_subject_summary(subject_summary_df)

    subject_channel_summary_df = build_subject_channel_summary(segment_df)

    # export_subject_outputs(
    #     subject_summary_df=subject_summary_df,
    #     subject_channel_summary_df=subject_channel_summary_df,
    #     output_dir=DatasetOverview_OutputDir_2_1,
    # )

    # -------------------------
    # 2.2.4 Similarity and candidate outliers
    # -------------------------
    similarity_df, distance_matrices = compute_activity_similarity(
        segment_df=segment_df,
        min_segments=3,
    )

    outlier_summary_df = summarize_candidate_outliers(similarity_df)
    print_similarity_summary(outlier_summary_df)
    print_top_candidate_outliers(similarity_df, top_n=70)

    export_similarity_outputs(
        similarity_df=similarity_df,
        outlier_summary_df=outlier_summary_df,
        output_dir=DatasetOverview_OutputDir_2_1,
        top_n=100,
    )

    # Build a unified manifest dataframe
    segment_manifest_df = segment_df.copy()

    segment_manifest_df["segment_id"] = segment_manifest_df.apply(
        lambda row: make_segment_id(
            row["experiment"], row["user"], row["activity"], row["start"], row["end"]
        ),
        axis=1,
    )

    if "avg_distance_to_class" in similarity_df.columns:
        similarity_merge_cols = [
            "experiment", "user", "activity", "start", "end",
            "avg_distance_to_class", "outlier_zscore", "candidate_outlier"
        ]

        segment_manifest_df = segment_manifest_df.merge(
            similarity_df[similarity_merge_cols],
            on=["experiment", "user", "activity", "start", "end"],
            how="left",
        )

    raw_recordings = load_raw_recordings(
        dataset_dir=DatasetDir,
        acc_pattern=ACC_PATTERN,
        gyro_pattern=GYRO_PATTERN,
    )

    segment_manifest_df, segment_data_dict, segment_indexes = build_segment_cache_from_manifest(
        segment_manifest_df=segment_manifest_df,
        raw_recordings=raw_recordings,
    )
    save_cache_bundle(
        cache_dir=DataCacheDir,
        manifest_df=segment_manifest_df,
        segment_data_dict=segment_data_dict,
        indexes=segment_indexes,
        raw_recordings=raw_recordings,
    )


#2.2.5
if Perform_PerSubjectAnalysis == 1:

    print(f"segment_manifest:  {segment_manifest_path}")
    print(f"segment_data_dict: {segment_data_dict_path}")
    print(f"segment_indexes:   {segment_indexes_path}")
    print(f"raw_recordings:    {raw_recordings_path}")

    segment_manifest_df = load_pickle(segment_manifest_path)
    segment_data_dict = load_pickle(segment_data_dict_path)
    segment_indexes = load_pickle(segment_indexes_path)

    base_df = segment_manifest_df.copy()

    subject_summary_df = summarize_subject_contribution(base_df)
    subject_channel_df = build_per_subject_channel_summary(base_df)

    print_subject_contribution_summary(subject_summary_df)

    # export_per_subject_outputs(
    #     subject_summary_df=subject_summary_df,
    #     subject_channel_df=subject_channel_df,
    #     output_dir=DatasetOverview_OutputDir_2_1,
    # )

    # Example variability bar plots
    # plot_subject_variability_bars(
    #     subject_channel_df=subject_channel_df,
    #     channel_stat="acc_z_std",
    #     save_dir=DatasetOverview_OutputDir_2_1,
    # )
    #
    # plot_subject_variability_bars(
    #     subject_channel_df=subject_channel_df,
    #     channel_stat="gyro_z_std",
    #     save_dir=DatasetOverview_OutputDir_2_1,
    # )

    # Compare same activity across subjects
    # Example: Walking
    # plot_same_activity_across_subjects(
    #     dataset_dir=DatasetDir,
    #     labels_file=LABELS_FILE,
    #     acc_pattern=ACC_PATTERN,
    #     gyro_pattern=GYRO_PATTERN,
    #     activity_id=4,  # WALKING
    #     max_subjects=5,
    #     channel_indices=(0, 1, 2),
    #     # save_dir=DatasetOverview_OutputDir_2_1,
    # )
    #
    # plot_same_activity_across_subjects_gyro(
    #     dataset_dir=DatasetDir,
    #     labels_file=LABELS_FILE,
    #     acc_pattern=ACC_PATTERN,
    #     gyro_pattern=GYRO_PATTERN,
    #     activity_id=1,  # WALKING
    #     max_subjects=5,
    #     channel_indices=(3, 4, 5),
    #     # save_dir=DatasetOverview_OutputDir_2_1,
    # )

    plot_per_subject_multibar(
        subject_channel_df=subject_channel_df,
        stat_type="mean",
        # save_dir=DatasetOverview_OutputDir_2_1,
    )

    plot_per_subject_multibar(
        subject_channel_df=subject_channel_df,
        stat_type="std",
        # save_dir=DatasetOverview_OutputDir_2_1,
    )

    plot_per_subject_multibar(
        subject_channel_df=subject_channel_df,
        stat_type="rms",
        # save_dir=DatasetOverview_OutputDir_2_1,
    )


if Perform_ChannelCorrelation == 1:
    segment_manifest_df = load_pickle(segment_manifest_path)
    base_df = segment_manifest_df.copy()

    # Overall correlation using segment means
    overall_corr_mean = compute_channel_correlation(
        segment_manifest_df=base_df,
        stat_type="std",
        group=None,
    )

    print_channel_correlation_summary(overall_corr_mean, label="Overall STD")

    plot_channel_correlation_heatmap(
        corr_df=overall_corr_mean,
        title="Overall Channel Correlation (STD)",
        # save_path=os.path.join(DatasetOverview_OutputDir_2_1, "overall_channel_correlation_mean.png"),
    )

    # Grouped correlation
    plot_grouped_channel_correlation(
        segment_manifest_df=base_df,
        stat_type="std",
        # save_dir=DatasetOverview_OutputDir_2_1,
    )


#2.3
# 2.3 Outlier Investigation and Segment Similarity Analysis
if Perform_OutlierInvestigation == 1:
    outlier_plot_dir = OutlierInvestigation_Dir

    for activity_id in sorted(similarity_df["activity"].unique()):
        segment_records = extract_activity_segments_with_metadata(
            dataset_dir=DatasetDir,
            labels_file=LABELS_FILE,
            acc_pattern=ACC_PATTERN,
            gyro_pattern=GYRO_PATTERN,
            target_activity=activity_id,
        )

        plot_activity_outliers_vs_typicals_acc(
            activity_id=activity_id,
            segment_records=segment_records,
            similarity_df=similarity_df,
            n_typical=3,
            save_dir=outlier_plot_dir,
        )
        #
        # plot_activity_outliers_vs_typicals_gyro(
        #     activity_id=activity_id,
        #     segment_records=segment_records,
        #     similarity_df=similarity_df,
        #     n_typical=3,
        #     save_dir=outlier_plot_dir,
        # )

        # plot_top_global_outliers(
        #     activity_id=activity_id,
        #     segment_records=segment_records,
        #     similarity_df=similarity_df,
        #     n_top_outliers=10,  # adjust as needed
        #     n_typical=5,
        #     save_dir=outlier_plot_dir,
        # )



