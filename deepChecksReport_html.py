from deepchecks.tabular.suites import data_integrity,train_test_validation
from deepchecks.tabular.checks.data_integrity import SpecialCharacters,StringLengthOutOfBounds,StringMismatch,\
    IsSingleValue,OutlierSampleDetection,DataDuplicates,MixedDataTypes,ClassImbalance,ColumnsInfo,ConflictingLabels,\
        FeatureLabelCorrelation,FeatureFeatureCorrelation,IdentifierLabelCorrelation,PercentOfNulls,MixedNulls
from deepchecks.tabular.checks.train_test_validation import WholeDatasetDrift,StringMismatchComparison,DatasetsSizeComparison,\
    TrainTestSamplesMix,DateTrainTestLeakageDuplicates,DateTrainTestLeakageOverlap,MultivariateDrift,TrainTestLabelDrift,\
        TrainTestFeatureDrift,CategoryMismatchTrainTest,FeatureLabelCorrelationChange,IdentifierLabelCorrelation as IdentifierLabelCorrelation2,\
            IndexTrainTestLeakage,NewLabelTrainTest
import io
import streamlit.components.v1 as components
from constants import *
def get_deepchecks_report_html(df,type,sub_type,df2=None):
    result=None
    if type=='data integrity train' or type=='data integrity test':
        check=data_integrity_dict[sub_type]
        result=check.run(df)
    elif type=='train test validation':
        check=train_test_validation_dict[sub_type]
        result=check.run(df,df2)
    if result is not None:
        string_io = io.StringIO()
        result.save_as_html(string_io)
        result_html = string_io.getvalue()
    return result_html


