import tempfile
import unittest
from pathlib import Path
from orbit.agents import AgentTeam
from orbit.tools import ToolRegistry

class AgentTeamTests(unittest.TestCase):
    def setUp(self):
        self.registry = ToolRegistry()
        self.registry.register("web_search", "web", lambda query, count=6: [{"title":"source","url":"https://example.com","query":query}], False)
        self.registry.register("search_files", "files", lambda query, relative=".": [{"path":"README.md","line":1,"text":query}], False)
        self.registry.register("calculate", "math", lambda expression: 4, False)

    def test_routes_independent_specialists(self):
        events=[]
        team=AgentTeam(self.registry, progress=lambda message, **details: events.append((message,details)))
        results=team.run("Research the latest vision models and search the project for image support")
        self.assertEqual({item["agent"] for item in results},{"researcher","workspace"})
        self.assertTrue(all(item["ok"] for item in results))
        self.assertTrue(any("parallel" in message for message,_ in events))

    def test_calculation_specialist_uses_safe_registered_tool(self):
        team=AgentTeam(self.registry)
        result=team.run("calculate 2 + 2")
        self.assertEqual(result[0]["agent"],"analyst")
        self.assertEqual(result[0]["result"],4)

    def test_irrelevant_requests_do_not_spawn_agents(self):
        self.assertEqual(AgentTeam(self.registry).run("Tell me a short story."),[])

    def test_specialist_failure_does_not_abort_other_agents(self):
        self.registry.register("bad_search", "not used", lambda: None, False)
        self.registry._tools["web_search"].handler=lambda **kwargs: (_ for _ in ()).throw(RuntimeError("offline"))
        results=AgentTeam(self.registry).run("Research current AI models and search the project for model")
        self.assertEqual(len(results),2)
        self.assertTrue(any(not item["ok"] for item in results))
        self.assertTrue(any(item["ok"] for item in results))

if __name__=="__main__":
    unittest.main()
