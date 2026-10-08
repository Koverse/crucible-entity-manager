'''
Custom functions for the object and event DataFrames
can be placed here. The name of the functions should be
placed in the configuration JSON file under the key
"custom_functions." The custom functions name needs
to be placed in the key value pair as follows for the
function "placeholder_function."

    "custom_functions":
        [
            {"function_name":"placeholder_function"}
        ]

The positional arguments for the function must be
    event_df        (DataFrame derived from SSE message)
    object_df       (DataFrame derived from object API)
    dataset_config  (dictionary derived from configuration JSON file for correlator)

'''

import json,time,os,sys,logging,uuid
import pandas as pd
import numpy as np

from typing import TypedDict, Tuple, Union



def placeholder_function(event_df,object_df,dataset_config):

    return event_df,object_df


def get_observation_time(event_df, object_df, dataset_config):
    created_date = event_df.get("crucibleHeader.createdDate")

    if {"now_ms", "seen_pos"}.issubset(event_df.columns):
        now_ms = pd.to_numeric(event_df["now_ms"], errors="coerce")
        seen_pos = pd.to_numeric(event_df["seen_pos"], errors="coerce")
        event_df["observation_time_ms"] = now_ms - seen_pos * 1000

        observed_at = pd.to_datetime(
            event_df["observation_time_ms"],
            unit="ms",
            utc=True,
            errors="coerce",
        )
        observation_iso = observed_at.dt.strftime("%Y-%m-%dT%H:%M:%S.%fZ")

        if created_date is not None:
            event_df["observation_time_iso"] = observation_iso.fillna(created_date)
        else:
            event_df["observation_time_iso"] = observation_iso
    elif created_date is not None:
        event_df["observation_time_iso"] = created_date
    else:
        raise KeyError("Expected now_ms/seen_pos or crucibleHeader.createdDate")

    return event_df, object_df

def format_crucible_header_uuid(event_df, object_df, dataset_config):
    column = 'crucibleHeader.uuid'
    if column not in event_df.columns:
        return event_df, object_df

    def _format_uuid(value):
        if value is None or (not isinstance(value, (list, tuple, np.ndarray)) and pd.isna(value)):
            return value
        if isinstance(value, uuid.UUID):
            return value.hex
        return uuid.UUID(str(value).strip()).hex

    event_df[column] = event_df[column].map(_format_uuid).to_numpy()
    return event_df, object_df

    
def fix_rtd_label(event_df, object_df, dataset_config):
    def _concat_label(v):
        if isinstance(v, (list, tuple, np.ndarray)):   # unwrap multi-valued cell
            parts = list(v)
        else:
            try:
                if pd.isna(v):
                    return None
            except (TypeError, ValueError):
                pass
            if v is None:
                return None
            parts = str(v).split(',')                   # also handles "1234,5678"
        # stringify, strip, drop blanks/NaN, then concat
        cleaned = []
        for p in parts:
            if p is None:
                continue
            try:
                if pd.isna(p):
                    continue
            except (TypeError, ValueError):
                pass
            s = str(p).strip()
            if s:
                cleaned.append(s)
        return ','.join(cleaned) if cleaned else None
    # Assign the raw array (.to_numpy()) rather than the mapped Series so pandas
    # does NOT try to align on the index. event_df can carry duplicate index
    # labels (concatenated SSE batches), and index alignment on assignment would
    # otherwise raise "cannot reindex on an axis with duplicate labels".
    event_df['intLabel'] = event_df['vehicle.vehicle.label'].map(_concat_label).to_numpy()
    return event_df, object_df
def get_velocity_open_sky(event_df,object_df,dataset_config):
    if 'velocity' in event_df.columns and 'true_track' in event_df.columns:
        event_df['northSpeed'] = event_df['velocity']*np.cos(np.radians(event_df['true_track']))
        event_df['eastSpeed'] = event_df['velocity']*np.sin(np.radians(event_df['true_track']))
    return event_df,object_df

def get_velocity(event_df,object_df,dataset_config):
    if 'gs' in event_df.columns and 'track' in event_df.columns:
        event_df['northSpeed'] = event_df['gs']*np.cos(np.radians(event_df['track']))
        event_df['eastSpeed'] = event_df['gs']*np.sin(np.radians(event_df['track']))
    return event_df,object_df

def set_measurement_errors(event_df,object_df,dataset_config):
    defaults = {
        'position_semimajor': 1000.,
        'position_semiminor': 700.,
        'position_orientation': 0.,
    }
    for column, default in defaults.items():
        if column not in event_df.columns:
            event_df[column] = default
        else:
            event_df[column] = event_df[column].fillna(default)
    return event_df,object_df

def drop_NaNs(event_df,object_df,dataset_config):
    if 'aircraft.model' not in event_df:
           event_df['aircraft.model' ] = 'UNKNOWN'
    event_df=event_df.dropna(subset=['lat','lon'])

    return event_df,object_df

def set_edh(event_df,object_df,dataset_config):

    # edhList=[ "CLS:S",]

    # object_df['edhControlSet']=''
    # object_df['edhControlSet']=object_df['edhControlSet'].apply(lambda x: edhList)

    edhList=[ "CLS:U",]
    event_df['defaultEDH']=''
    # event_df['defaultEDH']=event_df['defaultEDH'].apply(lambda x: edhList)
    event_df['defaultEDH']=[edhList for i in range(len(event_df))]
    return event_df,object_df


def set_description(event_df,object_df,dataset_config):
    # Build a human-readable identity.description from the synthetic
    # platform/route fields carried on each event (e.g. ADSB/AIS generators).
    # identity.description has a schema maxLength of 50, so the result is
    # truncated defensively.
    def _build_description(row):
        def _clean(val):
            # Treat None/NaN as empty (NaN is truthy, so `or ''` won't catch it).
            try:
                if pd.isna(val):
                    return ''
            except (TypeError, ValueError):
                pass
            return str(val).strip()

        platform = _clean(row.get('kinematicallyInferredPlatformType')) or 'UNKNOWN'
        origin = _clean(row.get('kinematicallyInferredOrigin')) or 'UNKNOWN'
        destination = _clean(row.get('kinematicallyInferredDestination')) or 'UNKNOWN'
        # Always include a default track identifier (ADSB hex / icao24, AIS MMSI)
        # so the description is never blank when route fields are missing.
        identifier = (_clean(row.get('hex')) or _clean(row.get('icao24'))
                      or _clean(row.get('MMSI')) or 'UNKNOWN')
        description = f'{platform} {origin}->{destination} ({identifier})'
        return description[:50]

    event_df['set_description'] = event_df.apply(_build_description, axis=1)
    return event_df,object_df




# old fix for kinematics in constructive data:

# def set_kinematics_nav_report(event_df,object_df,dataset_config):
#     event_ID_column='NavigationReport.MessageData.SystemID.DescriptiveLabel'
#     event_df = insert_values(
#         destination_df=event_df,
#         origin_df=object_df,
#         destination_ID_column=event_ID_column,
#         origin_ID_column='identity.callsign',
#         destination_column='set_lon',
#         origin_column='estimatedKinematics.position.longitude')
#     event_df = insert_values(
#         destination_df=event_df,
#         origin_df=object_df,
#         destination_ID_column=event_ID_column,
#         origin_ID_column='identity.callsign',
#         destination_column='set_lat',
#         origin_column='estimatedKinematics.position.latitude')
#     event_df = insert_values(
#         destination_df=event_df,
#         origin_df=object_df,
#         destination_ID_column=event_ID_column,
#         origin_ID_column='identity.callsign',
#         destination_column='set_alt',
#         origin_column='estimatedKinematics.position.altitude')
#     event_df = insert_values(
#         destination_df=event_df,
#         origin_df=object_df,
#         destination_ID_column=event_ID_column,
#         origin_ID_column='identity.callsign',
#         destination_column='set_kinematicsTimestamp',
#         origin_column='estimatedKinematics.kinematicsTimestamp')
    
#     return event_df,object_df
    

def set_kinematics_system_status(event_df,object_df,dataset_config):
    event_ID_column = 'SystemStatus.MessageData.SystemID.DescriptiveLabel'
    event_df = insert_values(
        destination_df=event_df,
        origin_df=object_df,
        destination_ID_column=event_ID_column,
        origin_ID_column='identity.callsign',
        destination_column='set_lon',
        origin_column='estimatedKinematics.position.longitude')
    event_df = insert_values(
        destination_df=event_df,
        origin_df=object_df,
        destination_ID_column=event_ID_column,
        origin_ID_column='identity.callsign',
        destination_column='set_lat',
        origin_column='estimatedKinematics.position.latitude')
    event_df = insert_values(
        destination_df=event_df,
        origin_df=object_df,
        destination_ID_column=event_ID_column,
        origin_ID_column='identity.callsign',
        destination_column='set_alt',
        origin_column='estimatedKinematics.position.altitude')
    event_df = insert_values(
        destination_df=event_df,
        origin_df=object_df,
        destination_ID_column=event_ID_column,
        origin_ID_column='identity.callsign',
        destination_column='set_kinematicsTimestamp',
        origin_column='estimatedKinematics.kinematicsTimestamp')    

    return event_df,object_df

# legacy entityStatus functions:
def object_status(object_df,dataset_config,row):
    if row['SystemStatus.MessageData.SystemState'] == 'FAILED':
        return 'DESTROYED'
    else:
        # leave default value by placing NaN, which is ignored
        return np.nan

def failed_destroyed(event_df,object_df,dataset_config):
    event_df['customStatus'] = event_df.apply (lambda row: object_status(object_df,dataset_config,row), axis=1)

    return event_df,object_df


##################
# Helper Functions
##################
    
def insert_values(destination_df=None, origin_df=None, destination_ID_column=None, origin_ID_column=None, destination_column=None, origin_column=None):
    """
    Insert values from origin_df into destination_df.
    
    Args:
        destination_df (DataFrame): The dataframe to insert values into.
        origin_df (DataFrame): The dataframe to copy values from.
        destination_ID_column (str): The column in destination_df to join on. 
        origin_ID_column (str): The column in origin_df to join on.
        destination_column (str): The column in destination_df to insert values into. 
        origin_column (str): The column in origin_df to copy values from.

        Note: kwargs were used to make the function easier to use in the context of the object_enrichment.py script. 

    Returns:
        DataFrame: Joined dataframe with values inserted.
    """
    
    # make sure that destination_column exists in destination_df
    if destination_column not in destination_df.columns:
        destination_df[destination_column] = np.nan


    joined_df = pd.merge(destination_df, origin_df, left_on=destination_ID_column, right_on=origin_ID_column, how='left', suffixes=(None, '__right'))
    # The origin_column might just be called 
    # origin_column 
    # in joined_df or it might be called
    # origin_column+'__right' 
    # if there was a name conflict. This is why we check both cases.

    if origin_column in joined_df.columns:
        joined_df[destination_column] = joined_df[origin_column]
    elif origin_column+'_right' in joined_df.columns:
        joined_df[destination_column] = joined_df[origin_column+'_right']
    else:
        logging.warning(f'Column {origin_column} not found in {origin_df}, continuing without inserting values.')
        return destination_df

    drop_col_list= [col for col in joined_df.columns if '__right' in col 
                    or (col in origin_df.columns and col not in destination_df.columns)]
    drop_col_list=[]

    for col in joined_df.columns:
        if ('__right' in col ) or (col in origin_df.columns and col not in destination_df.columns):
            drop_col_list.append(col)
                
    joined_df.drop(drop_col_list, axis=1, inplace=True)

    # make sure the destination_ID_column is a string
    joined_df[destination_ID_column] = joined_df[destination_ID_column].astype(str)


    return joined_df