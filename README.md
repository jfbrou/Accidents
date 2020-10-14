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

**Road Segments**

    - segment_id integer
    - class integer
    - direction integer
    - geometry geometry(LineString,32188)


**Accidents**

    - accident_id text
    - datetime timestamp
    - geometry geometry(Point,32188)


**Weather Records**


### SQL Queries

To find matching road segments / accidents

    SELECT accidents.accident_id as accident_id, road_segments.segment_id as segment_id, ST_Distance(accidents.geometry, ST_Centroid(road_segments.geometry)) as distance from accidents, road_segments where ST_Intersects(ST_Buffer(accidents.geometry, 100), road_segments.geometry) = true;


## Contributors

* **Jean-Felix Brouillette** - *Project Lead* - [jfbrou](https://github.com/jfbrou)

* **Jean-Romain Roy** - *Imagery Consultant* - [jeanromainroy](https://github.com/jeanromainroy)