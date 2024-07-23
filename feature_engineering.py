import pandas as pd
from sklearn.preprocessing import LabelEncoder


def create_features(use_all_columns: bool = True) -> pd.DataFrame:
    
    '''
    create features from the existing data 
    
    return: a pandas datafram with existing data + new features
    '''
    
    values_df = pd.read_csv('./data/train_values.csv')
    labels_df = pd.read_csv('./data/train_labels.csv')
    train_data = pd.merge(values_df, labels_df, on='building_id') 

    # rebin the age column
    cat_age = []
    for a in train_data['age']:
        if a <= 20:
            cat_age.append(0)
        elif a > 20 and a <= 75:
            cat_age.append(1)
        elif a > 75:
            cat_age.append(2)
    train_data['age_cat'] = cat_age
    
    #rebin the floor coutn: lump any value >=4 together (high-rise building) and keep the rest
    # as is (i.e. 1, 2, and 3)
    count_floors_pre_eq_ = [4 if f >= 4 else f for f in train_data['count_floors_pre_eq']]
    train_data['count_floors_pre_eq_rebin'] = count_floors_pre_eq_
    
    train_data['comb_types'] = train_data['foundation_type'].astype(str) + \
                            train_data['roof_type'].astype(str) + \
                            train_data['ground_floor_type'].astype(str) + \
                            train_data['other_floor_type'].astype(str)
                            # train_data['position'].astype(str) + \
                            # train_data['plan_configuration'].astype(str)
    
    train_data['comb_superstructure'] = train_data['has_superstructure_adobe_mud'].astype(str) + \
                                    train_data['has_superstructure_mud_mortar_stone'].astype(str) + \
                                    train_data['has_superstructure_stone_flag'].astype(str) + \
                                    train_data['has_superstructure_cement_mortar_stone'].astype(str) + \
                                    train_data['has_superstructure_mud_mortar_brick'].astype(str) + \
                                    train_data['has_superstructure_timber'].astype(str) + \
                                    train_data['has_superstructure_bamboo'].astype(str) + \
                                    train_data['has_superstructure_rc_non_engineered'].astype(str)
                                    # train_data['has_superstructure_other'].astype(str)
    
    train_data['comb_secondary_use'] = train_data['has_secondary_use'].astype(str) + \
                                    train_data['has_secondary_use_agriculture'].astype(str) + \
                                    train_data['has_secondary_use_hotel'].astype(str) + \
                                    train_data['has_secondary_use_rental'].astype(str) + \
                                    train_data['has_secondary_use_institution'].astype(str) + \
                                    train_data['has_secondary_use_school'].astype(str) + \
                                    train_data['has_secondary_use_industry'].astype(str) + \
                                    train_data['has_secondary_use_health_post'].astype(str) + \
                                    train_data['has_secondary_use_gov_office'].astype(str) + \
                                    train_data['has_secondary_use_use_police'].astype(str)
                                    # train_data['has_secondary_use_other'].astype(str)
    
    
    keep_columns = ['geo_level_1_id', 
                    'age_cat', 
                    'count_floors_pre_eq_rebin',
                    # 'count_floors_pre_eq',
                    'land_surface_condition',
                    'height_percentage', 
                    'area_percentage', 
                    'comb_types',
                    'position',
                    'plan_configuration',
                    'legal_ownership_status', 
                    'count_families', 
                    'comb_superstructure', 
                    'comb_secondary_use',
                    'damage_grade'
                    ]
    
    if use_all_columns:
        return train_data
    else:
        return train_data[keep_columns]


def encode_cat_features(train_data: pd.DataFrame, use_all_columns: bool = True) -> pd.DataFrame:
    '''
    take a dataframe with all sorts of data types
    encode the categorical features
    
    return: pandas datafram
    '''
    
    label_encoder = LabelEncoder()
    if use_all_columns:
        #encode categorical columns
        cat_cols = train_data.select_dtypes('object').columns

    else:
        cat_cols = ['land_surface_condition', 
                    'legal_ownership_status',
                    'comb_types', 
                    'position',
                    'plan_configuration',
                    'comb_superstructure', 
                    'comb_secondary_use'
                    ]

    for col in cat_cols:
        train_data[col] = label_encoder.fit_transform(train_data[col])
        
        
    return train_data