import random
from pathlib import Path
import os
import pickle
import io

import uvicorn
from fastapi.responses import StreamingResponse
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
from loguru import logger

from settings import Settings
from dijkstra_inference import DijkstraPath

origins = ["*"]

app = FastAPI(title=Settings().project_name)
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
BASE = Path(os.path.realpath(__file__)).parent
dijkstra = DijkstraPath(BASE / 'data/dijkstra.pickle', BASE / 'data/clear_nodes.pkl')
weights_dict = {}
for weight in (BASE / 'data/weights').iterdir():
    with open(weight, 'rb') as fd:
        weights_dict[weight.name.split('.')[0]] = pickle.load(fd)


class Points(BaseModel):
    start_lat: float = 55.7809453
    start_lon: float = 37.6373427
    end_lat: float = 55.6217188
    end_lon: float = 37.49859


class PointsTyped(BaseModel):
    start_lat: float = 55.7809453
    start_lon: float = 37.6373427
    end_lat: float = 55.6217188
    end_lon: float = 37.49859
    type_: str = "beauty_dist_weights"


@app.get('/')
def ping():
    return {'ping': 'pong'}


@app.post('/get_path')
def return_path(points: Points):
    """
    Возвращает три пути по двум заданным координатам
    Формат возвращаемых данных:
    """
    logger.info(points)
    if not all(map(lambda x: 0 <= x[1] <= 180, points)):
        raise HTTPException(status_code=400, detail="Every coordinate should be in 0..180")
    response = []
    for name, weights in weights_dict.items():
        path = dijkstra.get_shortest_path((points.start_lat, points.start_lon), (points.end_lat, points.end_lon),
                                          weights)
        dict_ = {'path': path[1][:-1], 'eta': None, 'type': name}
        response.append(dict_)
        #print(path[0])
    return response


@app.post('/get_path_csv/')
def return_path_csv(points: PointsTyped):
    """
    Возвращает csv по двум заданным координатам
    Возможные типы weights: beauty_dist, beauty, dist, hist, hist_dist, safety_dist, safety
    """
    shortest_path = dijkstra.get_shortest_path((points.start_lat, points.start_lon),
                                        (points.end_lat, points.end_lon), weights_dict[points.type_])[1]
    to_df = [(shortest_path[idx][0], shortest_path[idx][1], shortest_path[idx+1][0],
              shortest_path[idx+1][1]) for idx in range(len(shortest_path)-2)]
    df = pd.DataFrame(data=to_df, columns=['source_lat', 'source_lng', 'target_lat', 'target_lng'])

    stream = io.StringIO()
    df.to_csv(stream, index=False)
    response = StreamingResponse(iter([stream.getvalue()]), media_type="text/csv")
    response.headers["Content-Disposition"] = "attachment; filename=path.csv"
    return response


if __name__ == "__main__":
    app_name = "app:app"
    uvicorn.run(app_name, host=Settings().hostname, port=Settings().port, log_level="info", ssl_certfile=Settings().ssl_certfile, ssl_keyfile=Settings().ssl_keyfile)
