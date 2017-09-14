from cluster_programme import process_screen, merge_frames, partition_data
import pandas as pd 
su_df=pd.read_csv('../output/nomissing_survey.csv')
sc_df=pd.read_csv('../datafiles/fakescreen.csv')

#process screening data
sc_df=process_screen(sc_df)

#merge the dataset
merged=merge_frames(su_df,sc_df)

#partition the merged dataset
parts=partition_data(merged)

#print lenght of each partition
print len(parts[0]), len(parts[1]),len(parts[2])

val_data=parts[0]
test_data=parts[1]
training_data=parts[2]
print training_data.columns.values
#val_data.to_csv('val_data.csv')
#test_data.to_csv('test_data.csv')
#training_data.to_csv('training_data.csv')

merged.to_csv('../output/faketrain.csv', index=False)