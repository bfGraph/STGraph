"""Dataset loader provided by STGraph"""

from stgraph.dataset.static.cora_dataloader import CoraDataLoader

from stgraph.dataset.temporal.HungaryCPDataLoader import HungaryCPDataLoader
from stgraph.dataset.temporal.METRLADataLoader import METRLADataLoader
from stgraph.dataset.temporal.MontevideoBusDataLoader import MontevideoBusDataLoader
from stgraph.dataset.temporal.PedalMeDataLoader import PedalMeDataLoader
from stgraph.dataset.temporal.WikiMathDataLoader import WikiMathDataLoader
from stgraph.dataset.temporal.WindmillOutputDataLoader import WindmillOutputDataLoader

from stgraph.dataset.dynamic.england_covid_dataloader import EnglandCovidDataLoader
