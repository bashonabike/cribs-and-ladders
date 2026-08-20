import unittest
import unittest.mock as mock
from cribsandladders.EventSetBuilder import EventSetBuilder, ParamSet
from cribsandladders.Board import Board, Track
import numpy as np
import game_params as gp
import Enums as en

class TestEventSetBuilder(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures before each test method."""
        # Create a mock board with tracks
        self.mock_board = mock.MagicMock(spec=Board)
        self.mock_board.boardName = "tester"
        self.mock_board.boardID = 1
        self.mock_board.width = 0.0
        self.mock_board.height = 0.0
        self.mock_board.corners = None
        self.mock_board.tracks = []
        self.mock_board.twoDeckLineBoardPath = ""
        self.mock_board.possibleEvents = None

        self.track1 = mock.MagicMock(spec=Track)
        self.track1.Track_ID = 1
        self.track1.trackholes = [mock.MagicMock() for _ in range(10)]

        #TODO: build track object copy from actual test of events etc, spoof as needed

        self.track1.num = 0
        self.track1.length = 0
        self.track1.twodeckslength = 0
        self.track1.efflength = 0
        self.track1.ladders = []
        self.track1.chutes = []
        self.track1.eventsListLadder = []
        self.track1.eventsListChute = []
        self.track1.holesetfilepath = ""
        self.track1.holesetIndexer = []
        self.track1.candidateEvents = None
        self.track1.eventSetBuild = []
        self.track1.effLandingForHoles = []
        self.track1.instLocked = False
        # This is pointwise sum of event value (+/-) * likelihood of hit (1/length)
        # So sum of event values * # events / length
        # This will always be negative since always more chutes than ladders
        self.track1.simplEventImpedance = 0.0

        self.mock_board.tracks = [self.track1]
        
        # Mock possible events
        self.mock_possible_events = mock.MagicMock()
        self.builder = EventSetBuilder(self.mock_board, self.mock_possible_events)
        
        # Patch the random number generator for consistent tests
        self.rd_patch = mock.patch('random.random', return_value=0.5)
        self.mock_rand = self.rd_patch.start()
        
    def tearDown(self):
        """Clean up after each test method."""
        self.rd_patch.stop()
    
    def test_initialization(self):
        """Test that the EventSetBuilder initializes correctly."""
        self.assertEqual(self.builder.board, self.mock_board)
        self.assertEqual(self.builder.possibleEvents, self.mock_possible_events)
        self.assertEqual(len(self.builder.allTentLengthHisto), 0)
        self.assertIsInstance(self.builder.paramSet, ParamSet)
    
    def test_clear_event_set(self):
        """Test that clearEventSet resets all relevant attributes."""
        # Set some test values
        self.builder.allTentLengthHisto = [1, 2, 3]
        self.builder.orthos = 5
        self.builder.multis = 3
        self.builder.events = 10
        self.builder.cancels = 2
        self.builder.avgScoreSum = 100
        self.builder.avgScoreDiv = 10
        self.builder.avgScore = 10
        
        self.builder.clearEventSet()
        
        # Assert all values are reset
        self.assertEqual(len(self.builder.allTentLengthHisto), 0)
        self.assertEqual(self.builder.orthos, 0)
        self.assertEqual(self.builder.multis, 0)
        self.assertEqual(self.builder.events, 0)
        self.assertEqual(self.builder.cancels, 0)
        self.assertEqual(self.builder.avgScoreSum, 0)
        self.assertEqual(self.builder.avgScoreDiv, 0)
        self.assertEqual(self.builder.avgScore, 0)
    
    @mock.patch('cribsandladders.EventSetBuilder.cp.deepcopy')
    def test_optimize_setup(self, mock_deepcopy):
        """Test the optimizeSetup method."""
        # Mock the paramSet and tryEventSet
        self.builder.paramSet.monteCarlo = mock.MagicMock()
        self.builder.tryEventSet = mock.MagicMock(side_effect=[False, True])
        self.builder.buildSetIntoEvents = mock.MagicMock()
        
        # Setup mock for deepcopy
        mock_copy = mock.MagicMock()
        mock_deepcopy.return_value = mock_copy
        
        # Call the method
        self.builder.optimizeSetup()
        
        # Assertions
        self.builder.paramSet.monteCarlo.assert_called_once()
        self.assertEqual(self.builder.tryEventSet.call_count, 2)
        self.builder.buildSetIntoEvents.assert_called_once()
    
    def test_try_event_set(self):
        """Test the tryEventSet method."""
        # This is a complex method that would need more extensive mocking
        # We'll just test that it returns a boolean
        param_set = mock.MagicMock()
        prev_eff_lengths = [{'track_id': 1, 'efflength': 10}]
        
        # Mock the necessary methods
        # with mock.patch.object(self.builder, '_processTracks') as mock_process, \
        #      mock.patch.object(self.builder, '_calcTentativeEventSet') as mock_calc, \
        #      mock.patch.object(self.builder, '_evaluateEventSet') as mock_eval:
            
        # mock_eval.return_value = True
        result = self.builder.tryEventSet(param_set, prev_eff_lengths)

        self.assertTrue(result)
            # mock_process.assert_called_once()
            # mock_calc.assert_called_once()
            # mock_eval.assert_called_once()

class TestParamSet(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.mock_board = mock.MagicMock()
        self.mock_tracks = [mock.MagicMock(Track_ID=i) for i in range(1, 4)]
        self.param_set = ParamSet(self.mock_board, self.mock_tracks)
    
    def test_midpoint_init_params(self):
        """Test that midpointInitParams initializes parameters correctly."""
        self.param_set.midpointInitParams()
        self.assertGreater(len(self.param_set.params), 0)
        
        # Check that parameters are within their bounds
        for param in self.param_set.params:
            self.assertGreaterEqual(param['value'], param['min'])
            self.assertLessEqual(param['value'], param['max'])
    
    def test_monte_carlo(self):
        """Test that monteCarlo sets random values within bounds."""
        self.param_set.monteCarlo()
        self.assertGreater(len(self.param_set.params), 0)
        
        # Check that parameters are within their bounds
        for param in self.param_set.params:
            self.assertGreaterEqual(param['value'], param['min'])
            self.assertLessEqual(param['value'], param['max'])
    
    def test_try_get_param(self):
        """Test retrieving a parameter value."""
        # Add a test parameter
        test_param = {
            'track_ID': 1,
            'name': 'test_param',
            'value': 42,
            'min': 0,
            'max': 100
        }
        self.param_set.params = [test_param]
        
        # Test getting the parameter
        value = self.param_set.tryGetParam(1, 'test_param')
        self.assertEqual(value, 42)
        
        # Test getting non-existent parameter
        with self.assertRaises(Exception):
            self.param_set.tryGetParam(1, 'non_existent_param')
        
        # Test optional parameter that doesn't exist
        value = self.param_set.tryGetParam(1, 'non_existent_param', optional=True)
        self.assertIsNone(value)

if __name__ == '__main__':
    unittest.main()
