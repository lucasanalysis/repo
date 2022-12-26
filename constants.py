from deepchecks.tabular.suites import data_integrity,train_test_validation
from deepchecks.tabular.checks.data_integrity import SpecialCharacters,StringLengthOutOfBounds,StringMismatch,\
    IsSingleValue,OutlierSampleDetection,DataDuplicates,MixedDataTypes,ClassImbalance,ColumnsInfo,ConflictingLabels,\
        FeatureLabelCorrelation,FeatureFeatureCorrelation,IdentifierLabelCorrelation,PercentOfNulls,MixedNulls
from deepchecks.tabular.checks.train_test_validation import WholeDatasetDrift,StringMismatchComparison,DatasetsSizeComparison,\
    TrainTestSamplesMix,DateTrainTestLeakageDuplicates,DateTrainTestLeakageOverlap,MultivariateDrift,TrainTestLabelDrift,\
        TrainTestFeatureDrift,CategoryMismatchTrainTest,FeatureLabelCorrelationChange,IdentifierLabelCorrelation as IdentifierLabelCorrelation2,\
            IndexTrainTestLeakage,NewLabelTrainTest
import xgboost
import lightgbm as lgb
data_integrity_dict={'Full':data_integrity(),'SpecialCharacters':SpecialCharacters(),'StringLengthOutOfBounds':StringLengthOutOfBounds(),'StringMismatch':StringMismatch(),
    'IsSingleValue':IsSingleValue(),'OutlierSampleDetection':OutlierSampleDetection(),'DataDuplicates':DataDuplicates(),'MixedDataTypes':MixedDataTypes(),
    'ClassImbalance':ClassImbalance(),'ColumnsInfo':ColumnsInfo(),'ConflictingLabels':ConflictingLabels()
    ,'FeatureLabelCorrelation':FeatureLabelCorrelation(),'FeatureFeatureCorrelation':FeatureFeatureCorrelation(),
    'IdentifierLabelCorrelation':IdentifierLabelCorrelation(),'PercentOfNulls':PercentOfNulls(),'MixedNulls':MixedNulls()}
train_test_validation_dict={'Full':train_test_validation(),'WholeDatasetDrift':WholeDatasetDrift(),'StringMismatchComparison':StringMismatchComparison,
'DatasetsSizeComparison':DatasetsSizeComparison(),'TrainTestSamplesMix':TrainTestSamplesMix(),
'DateTrainTestLeakageDuplicates':DateTrainTestLeakageDuplicates(),'DateTrainTestLeakageOverlap':DateTrainTestLeakageOverlap(),
'MultivariateDrift':MultivariateDrift(),'TrainTestLabelDrift':TrainTestLabelDrift(),'TrainTestFeatureDrift':TrainTestFeatureDrift(),
'CategoryMismatchTrainTest':CategoryMismatchTrainTest(),'FeatureLabelCorrelationChange':FeatureLabelCorrelationChange(),
'IdentifierLabelCorrelation':IdentifierLabelCorrelation2(),'IndexTrainTestLeakage':IndexTrainTestLeakage(),
'NewLabelTrainTest':NewLabelTrainTest()}
data_integrity_choice=['Full','SpecialCharacters','StringLengthOutOfBounds','StringMismatch',
    'IsSingleValue','OutlierSampleDetection','DataDuplicates','MixedDataTypes','ClassImbalance','ColumnsInfo','ConflictingLabels',
    'FeatureLabelCorrelation','FeatureFeatureCorrelation','IdentifierLabelCorrelation','PercentOfNulls','MixedNulls']
train_test_validation_choice=['Full','WholeDatasetDrift','StringMismatchComparison','DatasetsSizeComparison',
    'TrainTestSamplesMix','DateTrainTestLeakageDuplicates','DateTrainTestLeakageOverlap','MultivariateDrift','TrainTestLabelDrift',
        'TrainTestFeatureDrift','CategoryMismatchTrainTest','FeatureLabelCorrelationChange','IdentifierLabelCorrelation',
            'IndexTrainTestLeakage','NewLabelTrainTest']
ml_model_dic={'XGBRegressor':xgboost.XGBRegressor(),'XGBClassifier':xgboost.XGBClassifier(),
'LGBMRegressor':lgb.LGBMRegressor(),'LGBMClassifier':lgb.LGBMClassifier()}