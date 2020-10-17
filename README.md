# Montreal Car Accident Risk Modeling

Our aim is to produce the best real-time, practical risk model for car accidents in Montreal.


## Environment

You will need,

 - Python >=3.6

 - PostgreSQL >=12.0

 - PostGIS >=3.0


## Dependencies

We suggest working out of a virtual environment

    mkdir venv

    python3 -m venv ./venv/

    source venv/bin/activate


Then installing the dependencies using pip,

    pip3 install -r requirements.txt


## Datatabase

We use PostgreSQL with the [PostGIS extension](https://postgis.net/).

### Data Model


**Accidents**

    - accident_id text
    - datetime timestamp
    - geometry geometry(Point,32188)
    - road_segment_id integer
    - weather jsonb


**Road Segments**

    - segment_id integer
    - class integer
    - direction integer
    - geometry geometry(LineString,32188)


**Weather Stations**

    - station_id text
    - geometry geometry(Point,32188)


**Weather Records**

    - index bigint
    - station_id text
    - datetime timestamp
    - temperature double precision
    - dewpoint double precision
    - humidity double precision
    - wdirection double precision
    - wspeed double precision
    - visibility double precision
    - pressure double precision
    - risky integer


### SQL Queries

To find matching road segments / accidents

    WITH acc_roadseg AS (
        WITH acci_roadseg_intersections AS (
            WITH accidents_subset AS (
                SELECT accident_id as accident_id, geometry as accident_geom FROM accidents LIMIT 30
            )
            SELECT accidents_subset.accident_id as accident_id,
                    road_segments.segment_id as road_segment_id,
                    accidents_subset.accident_geom as accident_geom,
                    ST_Distance(accidents_subset.accident_geom, ST_Centroid(road_segments.geometry)) as distance
            FROM accidents_subset, road_segments
            WHERE ST_Intersects(ST_Buffer(accidents_subset.accident_geom, 100), road_segments.geometry) = true
        )
        SELECT DISTINCT ON (accident_id) accident_id, road_segment_id, accident_geom
        FROM acci_roadseg_intersections ORDER BY accident_id, distance ASC
    )
    SELECT acc_roadseg.accident_id as accident_id,
            acc_roadseg.road_segment_id as road_segment_id,
            weather_stations.station_id as weather_station_id,
            ST_Distance(acc_roadseg.accident_geom, weather_stations.geometry) as weather_station_dist
    FROM acc_roadseg, weather_stations


## Data Features for training

With our source data we consolidate a dataset with the following features,

    - temperature
    - dewpoint
    - humidity
    - wdirection
    - wspeed
    - visibility
    - pressure

    - sun_elevation
    - sun_zenith
    - sun_azimuth


## Contributors

* **Jean-Felix Brouillette** - *Project Lead* - [jfbrou](https://github.com/jfbrou)

* **Jean-Romain Roy** - *Imagery Consultant* - [jeanromainroy](https://github.com/jeanromainroy)