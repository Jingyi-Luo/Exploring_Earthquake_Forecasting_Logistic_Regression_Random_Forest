# SYS6018 Final Project
# Jingyi Luo (jl6zh)

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
#from sklearn.metrics import roc_curve, auc
from obspy.clients.fdsn.client import Client
from obspy.core import UTCDateTime
#
working_dir = '/Users/ljyi/Desktop/SYS6018/final_project'
os.chdir(working_dir)

# =========================== Data Exploration 1. =============================
# prepare variables for data downloading
client = Client('USGS')
start_time = UTCDateTime("1990-01-01T00:00:00")
end_time = UTCDateTime("1991-01-01T00:00:00") # "2018-11-01T00:00:00"

# sent start time and end time to collect data
t1 = start_time
t2 = end_time

# # download earthquake data
#cat = client.get_events(starttime=t1, endtime=t2, minmagnitude=5.5)
cat = client.get_events(starttime=t1, endtime=t2, minmagnitude=4.5)

# save data into a easier to use format
grid_size = 5
lat_list = []
lon_list = []
mag_list = []
depth_list = []
grid_list = []
time_diff_list = []
for an_event in cat:
    lat = an_event.origins[0].latitude
    lon = an_event.origins[0].longitude
    depth = an_event.origins[0].depth * 0.001
    mag = an_event.magnitudes[0].mag
    origin_time = an_event.origins[0].time
    lat_list.append(lat)
    lon_list.append(lon)
    mag_list.append(mag)
    depth_list.append(depth)
    time_diff = origin_time - start_time
    time_diff_list.append(time_diff)
    grid_id = '{0}_{1}'.format(int(round(lat/grid_size))+0, int(round(lon/grid_size)))
    grid_list.append(grid_id)
# convert data to dataframe
data_df = pd.DataFrame(data={'lat': lat_list, 'lon': lon_list, 'mag': mag_list,
                             'depth': depth_list, 'time_diff': time_diff_list,
                             'grid_id':grid_list})
df = data_df
df.head(3) 
df.to_csv('processed_earthquake_data.csv', sep=',')

# check missing values 
df.isnull().sum(axis=0)
#lat          0
#lon          0
#mag          0
#depth        0
#time_diff    0
#grid_id      0

# There is no missing values for each column because only get the past data which 
# have earthquake. Then, transform these data in a easier use way.

# unique values for each column
#for col in df:
#    print(df[col].unique())
df.nunique()
#lat          524
#lon          523
#mag           25
#depth        335
#time_diff    526
#grid_id      190    

# distribution of each feature
#for col in df:
#    df.hist(column=col, edgecolor='black')
#    plt.savefig(col+'_hist.png')
# histgram for earthquake latitude
plt.hist(df['lat'], color='grey', edgecolor='black', bins=10)
plt.xlabel('Latitude Distribution of Earthquakes (magnitude>4.5)', fontsize=14)
plt.ylabel('Counts', fontsize=14)
plt.ylim(0,120)
#plt.show()
plt.savefig('lat_hist.png', dpi=300)
plt.close()

# histgram for earthquake longtitude
plt.hist(df['lon'], color='grey', edgecolor='black', bins=10)
plt.xlabel('Longtitude Distribution of Earthquakes (magnitude>4.5)', fontsize=14)
plt.ylabel('Counts', fontsize=14)
plt.ylim(0,140)
#plt.show()
plt.savefig('long_hist.png', dpi=300)
plt.close()

# histgram for earthquake magnitude
plt.hist(df['mag'], color='grey', edgecolor='black', bins=10)
plt.xlabel('Magnitude Distribution of Earthquakes (magnitude>4.5)', fontsize=14)
plt.ylabel('Counts', fontsize=14)
plt.ylim(0,300)
#plt.show()
plt.savefig('mag_hist.png', dpi=300)
plt.close()

# histgram for earthquake depth
plt.hist(df['depth'], color='grey', edgecolor='black', bins=10)
plt.xlabel('Depth Distribution of Earthquakes (magnitude>4.5)', fontsize=14)
plt.ylabel('Counts', fontsize=14)
plt.ylim(0,400)
#plt.show()
plt.savefig('depth_hist.png', dpi=300)
plt.close()

# time difference for earthquake depth
plt.hist(df['time_diff'], color='grey', edgecolor='black', bins=10)
plt.xlabel('Time Difference Distribution of Earthquakes', fontsize=14)
plt.ylabel('Counts', fontsize=14)
plt.ylim(0,70)
#plt.show()
plt.savefig('time_diff_hist.png', dpi=300)
plt.close()

# Most earthquakes happened around the equator.
# We collect the earthquake whose magnitude is larger than 4.5. Base on the 
# magnitude distribution, the magnitude of most earthquakes we collect range 
# from 5.5 to 6.5.
# The depth distribution shows the majority of the earthquakes's depth are less than
# 50 kilometers from ground, and some earthquakes' depths are between 50 kilometers 
# and 200 kilometers. The earthquakes happend below 400 kilometers from ground 
# are the least. The deepest distance was about 620 kilometers from ground.

# Check outliers    
# outliers
Q1 = df.quantile(0.25)
Q3 = df.quantile(0.75)
IQR = Q3 - Q1 
df = df.loc[:, ~df.columns.isin(['grid_id'])]
df.dtypes
#lat          float64
#lon          float64
#mag          float64
#depth        float64
#time_diff    float64
#grid_id       object

((df<(Q1-1.5*IQR)) | (df>(Q3+1.5*IQR))).sum()
#lat           2
#lon           0
#mag          28
#depth        66
#time_diff     0

# It seems that outliers existed. However, those 'outliers' are not outliers, 
# since earthquakes did happened at different depths with different magnitude.
    
# ===================== Data Transformation 2==================================
# extract latitude and longitude of earthquakes from data
# then assign each earthquake to a 5*5 grid on the map.
grid_size = 5
lat_list = []
lon_list = []
grid_list = []
for an_event in cat:
    lat = an_event.origins[0].latitude
    lon = an_event.origins[0].longitude
    lat_list.append(lat)
    lon_list.append(lon)
    grid_id = '{0}_{1}'.format(int(round(lat/grid_size))+0, int(round(lon/grid_size)))
    grid_list.append(grid_id)
    
# get unique grid 
unique_grid = list(set(grid_list))
unique_grid.sort()

# set forward and backward step to gather earthquake data
forward_time_step = 90*24*60*60 # 7 days
backward_time_step = 90*24*3600 # 30 days

# calculate the number of days collecting earthquake
ndays = int((end_time - start_time)/3600/24)

# set different combination of largest earthquake and earthquake counts
use_largest_earthquake = False
use_earthquake_count = True
# Summary:
# use_largest_earthquake:
#use_largest_earthquake = True
#use_earthquake_count = False

# use_earthquake_count:
#use_largest_earthquake = False
#use_earthquake_count = True

# use_largest_earthquake & use_earthquake_count:
#use_largest_earthquake = True
#use_earthquake_count = True
#
x_list = []
y_list = []

# group earthquake based on magnitude
def mag_count(x, x_min=5, x_max=6):
    larger_than = x['mag'].values >= x_min
    smaller_than = x['mag'].values < x_max
    return np.all([larger_than,smaller_than], axis=0)
 
# loop over ndays
for iday in range(ndays):
    print(iday)
    #
    t1 = start_time + iday*24*3600
    t2 = t1 + forward_time_step
    t3 = t1 - backward_time_step
    # get the earthquake in the future
    try:
        forward_cat = client.get_events(starttime=t1, endtime=t2, minmagnitude=5.5)
    except Exception as detail:
        forward_cat = None
        print('forward', iday, detail)
    # get the earthquake in the past
    try:
        backward_cat = client.get_events(starttime=t3, endtime=t1, minmagnitude=4.0)
    except Exception as detail:
        backward_cat = None
        print('backward', iday, detail)
    # past earthquakes: get the features means for each grid 
    if backward_cat is not None:
        backward_list = []
        for an_event in backward_cat:
            lat = an_event.origins[0].latitude
            lon = an_event.origins[0].longitude
            depth = an_event.origins[0].depth * 0.001
            mag = an_event.magnitudes[0].mag
            origin_time = an_event.origins[0].time
            time_diff = t1 - origin_time
            grid_id = '{0}_{1}'.format(int(round(lat/grid_size))+0, int(round(lon/grid_size)))
            temp_list = [lat, lon, depth, mag, time_diff, grid_id]
            backward_list.append(temp_list)
        backward_df_raw = pd.DataFrame(backward_list, columns=['lat', 'lon', 'depth', 'mag', 'time_diff', 'grid_id'])
        backward_df_group = backward_df_raw.groupby(['grid_id'])
        backward_df_mean = backward_df_group.mean()
        backward_df_mean.columns = [a+'_mean' for a in backward_df_mean.columns]
        # past earthquakes: get the largest earthquake for each grid
        if use_largest_earthquake:
            has_grid_id = list(set(backward_df_raw['grid_id'].values))
            largest_earthquake = []
            for a_grid_id in has_grid_id:
                subset = backward_df_raw[backward_df_raw['grid_id'] == a_grid_id]
                largest_index = np.argmax(subset['mag'].values)
                largest_earthquake.append(subset.iloc[largest_index].values)
            backward_df_largest = pd.DataFrame(largest_earthquake, columns=subset.columns.values,
                                               index=has_grid_id)
            # match column names to backward_df_largest: keep 'grid_id' as it, but joint '_largest' to other features' names
            column_names = []
            for a in backward_df_largest.columns:
                if a == 'grid_id':
                    column_names.append(a)
                else:
                    column_names.append(a+'_largest')
            backward_df_largest.columns = column_names
            #backward_df_largest.set_index(['grid_id'])
        # get past earthquake counts within certain magnitude range for each grid
        if use_earthquake_count:
            has_grid_id = list(set(backward_df_raw['grid_id'].values))
            min_mag_list = [4, 5, 6, 7]
            max_mag_list = [5, 6, 7, 10]
            earthquake_mag_count = []
            for a_grid_id in has_grid_id:
                subset = backward_df_raw[backward_df_raw['grid_id'] == a_grid_id]
                count_list = [a_grid_id]
                for i in range(len(min_mag_list)):
                    mag_min = min_mag_list[i]
                    mag_max = max_mag_list[i]
                    count = sum(mag_count(subset, x_min=mag_min, x_max=mag_max))
                    count_list.append(count)
                #
                earthquake_mag_count.append(count_list)
            # get column names for backward_df_mage_count
            backward_df_mag_count = pd.DataFrame(earthquake_mag_count, columns=['grid_id', 
                '4<mag<5', '5<mag<6','6<mag<7','7<mag<10'], index=has_grid_id)
    
            #backward_df_mag_count.set_index(['grid_id'])
        #backward_df_median = backward_df_group.median()
        #backward_df_median.columns = [a+'_median' for a in backward_df_median.columns]
        
        # get max and min earthquake value for each grid
        backward_df_max = backward_df_group.max()
        backward_df_max.columns = [a+'_max' for a in backward_df_max.columns]
        backward_df_min = backward_df_group.min()
        backward_df_min.columns = [a+'_min' for a in backward_df_min.columns]
        # different combinations of largest earthquake and earthquake counts for each grid
        if (not use_largest_earthquake) and (not use_earthquake_count):
            backward_df = pd.concat([backward_df_mean,backward_df_max, backward_df_min], axis=1)
        if use_largest_earthquake and (not use_earthquake_count):
            backward_df = pd.concat([backward_df_mean,backward_df_max, backward_df_min, backward_df_largest], axis=1)
        if (not use_largest_earthquake) and use_earthquake_count:
            backward_df = pd.concat([backward_df_mean,backward_df_max, backward_df_min, backward_df_mag_count], axis=1)
        if use_largest_earthquake and use_earthquake_count:
            backward_df = pd.concat([backward_df_mean,backward_df_max, backward_df_min, 
                                     backward_df_mag_count, backward_df_largest], axis=1)
        # drop extra 'grid_id'
        if 'grid_id' in backward_df.columns.values:
            backward_df = backward_df.drop(['grid_id'], axis=1)
        # future earthquake: get grid_id for each existing earthquake
        forward_list = []
        if forward_cat is not None:
            for an_event in forward_cat:
                lat = an_event.origins[0].latitude
                lon = an_event.origins[0].longitude
                grid_id = '{0}_{1}'.format(int(round(lat/grid_size))+0, int(round(lon/grid_size)))
                forward_list.append(grid_id)
        # save features to a sorted format
        feature_list = np.array([])
        feature_names = []
        for i in range(len(unique_grid)):
            grid_id = unique_grid[i]
            if grid_id in backward_df.index:
                feature_list = np.append(backward_df.loc[grid_id,].values, feature_list)
                feature_names = [grid_id+'_'+a for a in backward_df.columns] + feature_names
            else:
                feature_list = np.append(np.zeros(backward_df.shape[1]), feature_list)
                feature_names = [grid_id+'_'+a for a in backward_df.columns] + feature_names
        feature_names = ['grid_lat', 'grid_lon']+ feature_names
        x_feature = feature_list
        for i in range(len(unique_grid)):
            grid_id = unique_grid[i]
            words = grid_id.split('_')
            grid_lat = float(words[0])*grid_size
            grid_lon = float(words[1])*grid_size
            feature_list = np.append(np.array([grid_lat, grid_lon]), x_feature)
            x_list.append(feature_list)
            if grid_id in forward_list:
                y_list.append(1)
            else:
                y_list.append(0)
# save preprocessed data into a txt file
fid = open('features_tests/data_1990_mag_count_largest_earthquake.txt', 'w')
fid.write(', '.join(feature_names+['y']))
fid.write('\n')
for i in range(len(x_list)):
    fid.write(', '.join(str(a)for a in x_list[i]))
    fid.write(','+str(y_list[i]))
    fid.write('\n')
fid.close()

# ===================== Split and Scale Data for modeling =====================
from sklearn.preprocessing import StandardScaler

# read in data
#data = np.loadtxt('data_1990.txt', skiprows = 1, delimiter=',')
#x_list = data[:,:len(data[0])-1]
#y_list = data[:, -1].astype(int)

# 
x_train, x_test, y_train, y_test = train_test_split(x_list, y_list, test_size=0.2, random_state=1)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=1)

# feature scaling
sc = StandardScaler()
x_train = sc.fit_transform(x_train) 
x_val = sc.transform(x_val) 
x_test = sc.transform(x_test)

# =========================== Logistic Regression =============================
from sklearn.metrics import roc_auc_score

# training model
log_clf = LogisticRegression(random_state=0, solver='lbfgs')
log_clf.fit(x_train, y_train)

# validate 
y_val_log = log_clf.predict(x_val)
y_val_prob_log = log_clf.predict_proba(x_val)[:, 1]
val_auc_log = roc_auc_score(y_val, y_val_prob_log)
print('Logistic validation roc: {0:8.3f}'.format(val_auc_log))
# Year (1990.1.1 - 1991.1.1)  0.549

# predict
y_pred_log = log_clf.predict(x_test)
y_pred_prob_log = log_clf.predict_proba(x_test)[:, 1]
auc_log = roc_auc_score(y_test, y_pred_prob_log)
print('Logistic prediction roc: {0:8.3f}'.format(auc_log))
# Year (1990.1.1 - 1991.1.1)  0.522

# =========================== Random Forest ===================================
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

# training model
RF = RandomForestClassifier(n_estimators=100, n_jobs = -1, random_state=0)
RF.fit(x_train, y_train)

# validate
y_val_RF = RF.predict(x_val)
y_val_prob_RF = RF.predict_proba(x_val)[:, 1]
val_auc_RF = roc_auc_score(y_val, y_val_prob_RF)
print('RF validation roc: {0:8.3f}'.format(val_auc_RF))
# Year (1990.1.1 - 1991.1.1)  0.582

# prediction
y_pred_RF = RF.predict(x_test)
y_pred_prob_RF = RF.predict_proba(x_test)[:, 1]
auc_RF = roc_auc_score(y_test, y_pred_prob_RF)
print('RF prediction roc: {0:8.3f}'.format(auc_RF))
# Year (1990.1.1 - 1991.1.1)  0.563







