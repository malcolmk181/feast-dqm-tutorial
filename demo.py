#!/usr/bin/env python
# coding: utf-8

# In[5]:


get_ipython().run_line_magic('pip', "install 'feast[ge]'")


# In[6]:


import pyarrow.parquet
import pandas as pd

from feast import FeatureView, Entity, FeatureStore, Field, BatchFeatureView
from feast.types import Float64, Int64
from feast.value_type import ValueType
from feast.data_format import ParquetFormat
from feast.on_demand_feature_view import on_demand_feature_view
from feast.infra.offline_stores.file_source import FileSource
from feast.infra.offline_stores.file import SavedDatasetFileStorage
from datetime import timedelta


# ## Declare features

# In[7]:


# create FileSource
batch_source = FileSource(
    timestamp_field="day",
    path="trips_stats.parquet",  # using parquet file included from repo
    file_format=ParquetFormat()
)


# In[8]:


# create taxi entity
taxi_entity = Entity(name='taxi', join_keys=['taxi_id'])


# In[9]:


# create feature view
trips_stats_fv = BatchFeatureView(
    name='trip_stats',
    entities=[taxi_entity],
    schema=[
        Field(name="total_miles_travelled", dtype=Float64),
        Field(name="total_trip_seconds", dtype=Float64),
        Field(name="total_earned", dtype=Float64),
        Field(name="trip_count", dtype=Int64),

    ],
    ttl=timedelta(seconds=86400),
    source=batch_source,
)


# In[10]:


# create on-demand feature view
# these are also still in alpha iirc, and don't scale well
@on_demand_feature_view(
    sources=[
      trips_stats_fv,
    ],
    schema=[
        Field(name="avg_fare", dtype=Float64),
        Field(name="avg_speed", dtype=Float64),
        Field(name="avg_trip_seconds", dtype=Float64),
        Field(name="earned_per_hour", dtype=Float64),
    ]
)
def on_demand_stats(inp: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame()
    out["avg_fare"] = inp["total_earned"] / inp["trip_count"]
    out["avg_speed"] = 3600 * inp["total_miles_travelled"] / inp["total_trip_seconds"]
    out["avg_trip_seconds"] = inp["total_trip_seconds"] / inp["trip_count"]
    out["earned_per_hour"] = 3600 * inp["total_earned"] / inp["total_trip_seconds"]
    return out


# In[23]:


store = FeatureStore(fs_yaml_file="feature_store.yaml")  # using feature_store.yaml that stored in the same directory
store.project


# In[20]:


store.apply([taxi_entity, trips_stats_fv, on_demand_stats])  # writing to the registry


# ## Generate training reference dataset

# In[13]:


taxi_ids = pyarrow.parquet.read_table("entities.parquet").to_pandas()


# In[14]:


# generate range of timestamps with daily frequency
timestamps = pd.DataFrame()
timestamps["event_timestamp"] = pd.date_range("2019-06-01", "2019-07-01", freq='D')


# In[15]:


# Cross merge (aka relation multiplication) produces entity dataframe with each taxi_id repeated for each timestamp:
entity_df = pd.merge(taxi_ids, timestamps, how='cross')
entity_df


# In[22]:


# Retrieving historical features for resulting entity dataframe and persisting output as a saved dataset:
job = store.get_historical_features(
    entity_df=entity_df,
    features=[
        "trip_stats:total_miles_travelled",
        "trip_stats:total_trip_seconds",
        "trip_stats:total_earned",
        "trip_stats:trip_count",
        "on_demand_stats:avg_fare",
        "on_demand_stats:avg_trip_seconds",
        "on_demand_stats:avg_speed",
        "on_demand_stats:earned_per_hour",
    ]
)

# this is the new thing!
store.create_saved_dataset(
    from_=job,
    name='my_training_ds',
    storage=SavedDatasetFileStorage(path='my_training_ds.parquet')
)


# In[3]:


saved_dataset = pyarrow.parquet.read_table("my_training_ds.parquet").to_pandas()


# In[4]:


saved_dataset


# ## Develop dataset profiler
# 
# The dataset profiler is a function that accepts a dataset and generates the set of its characteristics. These characteristics will then be used to evaluate (validate) other datasets.
# 
# Important: datasets are not compared to each other! Feast uses a reference dataset and a profiler function to generate a reference profile. This profile will then be used during validation of the tested dataset.

# In[17]:


import numpy as np

from feast.dqm.profilers.ge_profiler import ge_profiler

from great_expectations.core.expectation_suite import ExpectationSuite
from great_expectations.dataset import PandasDataset


# In[20]:


# Load saved dataset from earlier and show contents
ds = store.get_saved_dataset('my_training_ds')
ds.to_df()


# Feast uses Great Expectations as a validation engine and ExpectationSuite as a dataset's profile. Hence, we need to develop a function that will generate an ExpectationSuite.
# 
# This function will receive an instance of PandasDataset (a wrapper around pandas.DataFrame) so we can utilize both the Pandas DataFrame API and some helper functions from PandasDataset during profiling.

# In[21]:


DELTA = 0.1  # controlling allowed window in fraction of the value on scale [0, 1]

@ge_profiler # decorator from great expectations
def stats_profiler(ds: PandasDataset) -> ExpectationSuite:
    # simple checks on data consistency
    ds.expect_column_values_to_be_between(
        "avg_speed",
        min_value=0,
        max_value=60,
        mostly=0.99  # allow some outliers
    )

    ds.expect_column_values_to_be_between(
        "total_miles_travelled",
        min_value=0,
        max_value=500,
        mostly=0.99  # allow some outliers
    )

    # expectation of means based on observed values
    observed_mean = ds.trip_count.mean()
    ds.expect_column_mean_to_be_between("trip_count",
                                        min_value=observed_mean * (1 - DELTA),
                                        max_value=observed_mean * (1 + DELTA))

    observed_mean = ds.earned_per_hour.mean()
    ds.expect_column_mean_to_be_between("earned_per_hour",
                                        min_value=observed_mean * (1 - DELTA),
                                        max_value=observed_mean * (1 + DELTA))


    # expectation of quantiles
    qs = [0.5, 0.75, 0.9, 0.95]
    observed_quantiles = ds.avg_fare.quantile(qs)

    ds.expect_column_quantile_values_to_be_between(
        "avg_fare",
        quantile_ranges={
            "quantiles": qs,
            "value_ranges": [[None, max_value] for max_value in observed_quantiles]
        })

    return ds.get_expectation_suite()


# In[23]:


# check out profile of saved dataset
ds.get_profile(profiler=stats_profiler)


# In[25]:


# Now we can create validation reference from training dataset and profiler function:
validation_reference = ds.as_reference(name="validation_reference_dataset", profiler=stats_profiler)


# In[28]:


# test against existing retrieval job (which was pulling training dataset)
_ = job.to_df(validation_reference=validation_reference)
# validation passes if no exception is raised


# ## Validate New Historical Retrieval

# In[29]:


from feast.dqm.errors import ValidationFailed


# In[30]:


# Create some new timestamps for Dec 2020
timestamps = pd.DataFrame()
timestamps["event_timestamp"] = pd.date_range("2020-12-01", "2020-12-07", freq='D')
entity_df = pd.merge(taxi_ids, timestamps, how='cross')
entity_df


# In[31]:


# pull the feature for the Dec 2020 timestamps
job = store.get_historical_features(
    entity_df=entity_df,
    features=[
        "trip_stats:total_miles_travelled",
        "trip_stats:total_trip_seconds",
        "trip_stats:total_earned",
        "trip_stats:trip_count",
        "on_demand_stats:avg_fare",
        "on_demand_stats:avg_trip_seconds",
        "on_demand_stats:avg_speed",
        "on_demand_stats:earned_per_hour",
    ]
)


# In[32]:


# Try to convert the pulled feature to a dataframe AND VALIDATE
try:
    df = job.to_df(validation_reference=validation_reference)
except ValidationFailed as exc:
    print(exc.validation_report)


# The validation failed since several expectations didn't pass:
# - Trip count (mean) decreased more than 10% (which is expected when comparing Dec 2020 vs June 2019)
# - Average Fare increased - all quantiles are higher than expected
# - Earn per hour (mean) increased more than 10% (most probably due to increased fare)

# Note that an exception was raised, which is an easy way to stop a pipeline!
# 
# The exception also returns (via `.validation_report`) a json list with the errors. This json can easily be saved to a file, parsed into a slack message, or whatever.
