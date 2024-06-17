import pandas as pd
import numpy as np
import copy
import json
import geopandas as gpd
import shapely
from shapely.geometry import box
from shapely.ops import split
from arcgis.gis import GIS
from arcgis import geocode
from arcgis.geometry import BaseGeometry, Geometry
import openai
from keybert.llm import OpenAI
from keybert import KeyLLM


@classmethod
def from_shapely(cls, shapely_geometry):
    return cls(shapely_geometry.__geo_interface__)

BaseGeometry.from_shapely = from_shapely


# init GIS connection
with open("agol_account_info.txt", "r") as f:
    url, username, password = f.read().splitlines()
    
gis = GIS(url, username=username, password=password)

# init llm connection
with open("openai_api_key", "r") as f:
    MY_API_KEY = f.readline()
    
    
# Create your LLM
client = openai.OpenAI(api_key=MY_API_KEY)
llm = OpenAI(client)

# Load it in KeyLLM
kw_model = KeyLLM(llm)



stop_words = ['region', 'location', 'geographical location', 'street', 'streets', 'landmarks', 'area', 'part']
directional_predictates = ['north', 'south', 'west', 'east']
arcgis_community_boundaries_lyr = gis.content.get("23a806fb906e428cb75d123cf2ab580c").layers[0]
community_boundaries_sdf = pd.DataFrame.spatial.from_layer(arcgis_community_boundaries_lyr)
fset = arcgis_community_boundaries_lyr.query()
gjson_string = fset.to_geojson
community_boundaries_gdf = gpd.read_file(gjson_string, driver='GeoJSON').set_crs(2230, allow_override=True)
sd_roads_gdf = gpd.read_file('sd_roads.json', driver='GeoJSON').set_crs(2230, allow_override=True)



def get_extent(geom_id, geom_txt):
    if geom_id == 0:
        return get_community(geom_txt)
    elif geom_id == 1:
        return get_poi(geom_txt)
    elif geom_id == 2:
        return get_road(geom_txt)
    else:
        raise Exception("Invalid geom_id")


def get_community(geom_txt):
    
    keywords = kw_model.extract_keywords(geom_txt)[0]
    dw_out = []
    community_out = []
    for kw in copy.deepcopy(keywords):
        for dw in directional_predictates:
            if dw in kw.lower():
                dw_out.append(dw)
                keywords.remove(kw)


    for kw in keywords:    
        kw_community = community_neighborhoods_df[
            community_neighborhoods_df.apply(
                lambda row: kw.lower() in row['community'].lower(), 
                axis = 1)
        ]
        if len(kw_community):
            community_out.append(kw)

    geocode_out = []
    for kw in community_out:
        geocode_out.append(geocode(address = kw, max_locations = 10))
        
        
    for g_out in geocode_out[0]:
        p = g_out['location']
        p = gpd.GeoSeries(shapely.Point(p['x'],  p['y'])).set_crs(4326).to_crs(2230)
        temp_gdf = community_boundaries_gdf[community_boundaries_gdf.geometry.contains(p[0])]
        if len(temp_gdf):
            # the geocode out are ranked in confidence score
            # break the first outcome is found
            break
    
    community_shape = temp_gdf.geometry.iloc[0]
    x,y = community_shape.centroid.x, community_shape.centroid.y
    minx, miny, maxx, maxy = community_shape.bounds
    
    
    l1 = shapely.LineString([(x, miny), (x, maxy)])
    l2 = shapely.LineString([(minx, y), (maxx, y)])
    
    out_poly = []
    for dw in dw_out:
        if dw in ['north', 'south']:
            out_dict = dict(map(lambda i,j : (i,j) , ['south', 'north'], split(community_shape, l2).geoms))
        else:
            out_dict = dict(map(lambda i,j : (i,j) , ['south', 'north'], split(community_shape, l2).geoms))
        out_poly.append(out_dict[dw])
        
    return out_poly[0]
    

def get_poi(geom_txt):
    
    keywords = kw_model.extract_keywords(e)[0]
    for kw in keywords.copy():
        if kw.lower() in stop_words:
            keywords.remove(kw)
            
    geocode_out = geocode(address = " ".join(keywords) + ", San Diego", max_locations = 1)[0]
    return gpd.GeoSeries(shapely.Point(geocode_out['location']['x'],  geocode_out['location']['y'])).set_crs(4326).to_crs(2230).buffer(100)[0]


def get_road(geom_txt):
    keywords = kw_model.extract_keywords(geom_txt)[0]
    for kw in keywords.copy():
        if kw.lower() in stop_words:
            keywords.remove(kw)
            
    geocode_out = []
    # specify SD to be disambiguous for geocode
    for kw in keywords:
        geocode_out.append(geocode(address = kw + ", San Diego", max_locations = 1)[0]['location'])
    
    
    rd = []
    for g_out in geocode_out:
        p = gpd.GeoSeries(shapely.Point(g_out['x'],  g_out['y'])).set_crs(4326).to_crs(2230).buffer(100)
        rd.append(sd_roads_gdf[sd_roads_gdf.geometry.crosses(p[0])])
    
    out_road = []
    for r in rd:
        rd_name = r.RD20FULL.iloc[0]
        road_segments = sd_roads_gdf[sd_roads_gdf['RD20FULL'] == rd_name]['geometry'].tolist()
        temp_road = road_segments[0]
        for r in road_segments[1:]:
            temp_road = temp_road.union(r)
        out_road.append(temp_road)
        
    simple_rd = []
    for rd in out_road:
        minx, miny, maxx, maxy = rd.bounds
        if (maxy - miny) > (maxx - minx):
            x = rd.centroid.x
            simple_rd.append(shapely.LineString([(x, miny), (x, maxy)]))
        else:

            y = rd.centroid.y
            simple_rd.append(shapely.LineString([(minx, y), (maxx, y)]))
    
    
    rd_bound = []
    road = None
    for rd1 in simple_rd:
        count = 0
        for rd2 in simple_rd:
            if rd1.crosses(rd2): count += 1 
        if count == 2:
            for i, rd2 in enumerate(simple_rd):
                pt = rd1.intersection(rd2)
                if isinstance(pt, shapely.Point):
                    rd_bound.append(pt)
                else:
                    road = out_road[i]
            break
            
    idxmax = np.argmax(np.abs(np.array(rd_bound[0].bounds) - rd_bound[1].bounds))
    maxpt1, maxpt2 = rd_bound[0].xy[idxmax][0], rd_bound[1].xy[idxmax][0]
    minx, miny, maxx, maxy = road.bounds
    minx, maxx = min(maxpt1, maxpt2), max(maxpt1, maxpt2)
    
    out_road = box(minx, miny, maxx, maxy).intersection(road)
    
    return out_road.buffer(100)