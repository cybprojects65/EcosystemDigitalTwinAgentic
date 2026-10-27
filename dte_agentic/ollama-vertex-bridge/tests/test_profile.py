import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location('bridge', Path(__file__).resolve().parents[1] / 'ollama_vertex_bridge_basic_auth.py')
bridge = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bridge)

class ProfileTests(unittest.TestCase):
    def setUp(self):
        self.settings = patch.multiple(bridge.CONFIG, default_max_tokens=1536, default_temperature=0.0, local_context_window=4096)
        self.settings.start()
        self.addCleanup(self.settings.stop)

    def test_generate_and_chat_override_stale_client_options(self):
        options = {'num_predict': 2500, 'temperature': 0.8, 'num_ctx': 16384, 'seed': 42}
        for payload in (bridge.generate_to_vertex({'prompt': 'Question', 'system': 'Instructions', 'options': options}),
                        bridge.chat_to_vertex({'messages': [{'role': 'system', 'content': 'Instructions'}, {'role': 'user', 'content': 'Question'}], 'options': options})):
            self.assertEqual(payload['generationConfig']['maxOutputTokens'], 1536)
            self.assertEqual(payload['generationConfig']['temperature'], 0)
            self.assertEqual(payload['generationConfig']['seed'], 42)
            self.assertNotIn('num_ctx', payload['generationConfig'])
            with patch.object(bridge, 'run_vertex_request', return_value=([{'totalTokens': 2560}], 0)) as remote:
                self.assertEqual(bridge.enforce_context_budget(payload), 2560)
                args, kwargs = remote.call_args
                self.assertTrue(kwargs['url'].endswith(':countTokens'))
                self.assertEqual(args[0]['systemInstruction'], payload['systemInstruction'])
                self.assertEqual(args[0]['contents'], payload['contents'])

    def test_overflow_rejected(self):
        payload = bridge.generate_to_vertex({'prompt': 'Question'})
        with patch.object(bridge, 'run_vertex_request', return_value=([{'totalTokens': 2561}], 0)):
            with self.assertRaises(bridge.BridgeError) as error:
                bridge.enforce_context_budget(payload)
            self.assertEqual(error.exception.status, 400)

    def test_invalid_count_rejected(self):
        for response in ([], [{}], [{'totalTokens': -1}], [{'totalTokens': '10'}]):
            with patch.object(bridge, 'run_vertex_request', return_value=(response, 0)):
                with self.assertRaises(bridge.BridgeError):
                    bridge.enforce_context_budget(bridge.generate_to_vertex({'prompt': 'Question'}))

    def test_count_failure_not_bypassed(self):
        with patch.object(bridge, 'run_vertex_request', side_effect=bridge.BridgeError('unavailable', 502)):
            with self.assertRaises(bridge.BridgeError):
                bridge.enforce_context_budget(bridge.generate_to_vertex({'prompt': 'Question'}))

if __name__ == '__main__':
    unittest.main()
