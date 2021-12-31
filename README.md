FeatureCreator:
- apply_feature - list of columns to apply trasnformations
- save_feature - column to save values, if absent will saved to apply_feature columns
- use_just_for_train - feature creator will used on train step
- creator_apply_type - Enum with feature applying rules
  - BeforeTrainBeforeLabel - applying before train and before labeling creator
  - BeforeTrainAfterLabel - applying before train and before labeling creator
  - OnceOnAllDataBeforeLabel - applying once and before labeling creator
  - OnceOnAllDataAfterLabel - applying once and after labeling creator
