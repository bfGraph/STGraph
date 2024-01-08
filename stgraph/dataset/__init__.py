"""Dataset loader provided by STGraph"""

from stgraph.dataset.STGraphDataset import STGraphDataset

from stgraph.dataset.static.STGraphStaticDataset import STGraphStaticDataset
from stgraph.dataset.static.CoraDataLoader import CoraDataLoader

from stgraph.dataset.temporal.STGraphTemporalDataset import STGraphTemporalDataset
from stgraph.dataset.temporal.HungaryCPDataLoader import HungaryCPDataLoader
from stgraph.dataset.temporal.METRLADataLoader import METRLADataLoader
from stgraph.dataset.temporal.MontevideoBusDataLoader import MontevideoBusDataLoader
from stgraph.dataset.temporal.PedalMeDataLoader import PedalMeDataLoader
from stgraph.dataset.temporal.WikiMathDataLoader import WikiMathDataLoader
from stgraph.dataset.temporal.WindmillOutputDataLoader import WindmillOutputDataLoader

from stgraph.dataset.dynamic.STGraphDynamicDataset import STGraphDynamicDataset
from stgraph.dataset.dynamic.EnglandCovidDataLoader import EnglandCovidDataLoader
