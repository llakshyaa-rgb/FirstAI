import unittest
from orbit.planner import parse_tool_plan, execute_tool_plan
from orbit.tools import ToolRegistry

class PlannerTests(unittest.TestCase):
    def setUp(self):
        self.registry = ToolRegistry()
        self.registry.register("calculate", "safe math", lambda expression: 2 + 2, requires_confirmation=False)
        self.registry.register("write_file", "writes", lambda: "written", requires_confirmation=True)

    def test_valid_plan_executes_only_registered_safe_tool(self):
        raw = '{"steps":[{"tool":"calculate","arguments":{"expression":"2+2"}}]}'
        plan = parse_tool_plan(raw, self.registry)
        self.assertEqual(plan[0][0], "calculate")
        result = execute_tool_plan(raw, self.registry)
        self.assertTrue(result[0]["ok"])
        self.assertEqual(result[0]["result"], 4)

    def test_write_tools_cannot_be_called_by_model_plan(self):
        raw = '{"steps":[{"tool":"write_file","arguments":{}}]}'
        self.assertIsNone(parse_tool_plan(raw, self.registry))

    def test_unknown_tool_and_extra_fields_are_rejected(self):
        self.assertIsNone(parse_tool_plan('{"steps":[{"tool":"shell","arguments":{}}]}', self.registry))
        self.assertIsNone(parse_tool_plan('{"steps":[],"execute_anything":true}', self.registry))

    def test_malformed_json_is_ignored(self):
        self.assertIsNone(parse_tool_plan("not json", self.registry))

if __name__ == "__main__":
    unittest.main()
