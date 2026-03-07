"""專案管理相關方法。"""
from __future__ import annotations

from textual import work
from textual.containers import Vertical


class ProjectMixin:
    """專案管理相關方法。需要混入 App 才能使用。"""

    def action_open_project_modal(self) -> None:
        self._load_and_show_project_modal()

    @work(exclusive=True)
    async def _load_and_show_project_modal(self) -> None:
        from modals import ProjectModal
        projects = await self._project_manager.list_projects()
        current_id = self.state.current_project["id"] if self.state.current_project else None

        def on_dismiss(result) -> None:
            if result is None:
                return
            action, data = result
            if action == 'create':
                self._create_and_switch_project(data)
            elif action == 'switch':
                self._switch_project(data)

        self.app.push_screen(ProjectModal(projects, current_id=current_id), callback=on_dismiss)

    @work(exclusive=True)
    async def _create_and_switch_project(self, name: str) -> None:
        project = await self._project_manager.create_project(name)
        await self._do_switch_project(project)

    @work(exclusive=True)
    async def _switch_project(self, project: dict) -> None:
        await self._do_switch_project(project)

    async def _do_switch_project(self, project: dict) -> None:
        import gc
        from paths import project_root, ensure_project_dirs
        from conversation.store import ConversationStore
        from rag.knowledge_graph import reset_rag
        from widgets import WelcomeBlock

        self.state.current_project = project
        self.state.current_conversation_id = None
        pid = project["id"]

        ensure_project_dirs(pid)

        proj_root = project_root(pid)
        self._conv_store = ConversationStore(db_path=proj_root / "conversations.db")

        await self._project_manager.touch_project(pid)
        reset_rag()
        gc.collect()
        self.messages = []
        container = self.query_one("#messages", Vertical)
        await container.remove_children()
        welcome = WelcomeBlock()
        welcome.border_title = "[#d4a27a]◈[/] De-insight v0.5"
        await container.mount(welcome)
        self._refresh_memory_panel()
        self._refresh_knowledge_panel()
        self._update_menu_bar()
        self.notify(f"已切換到：{project['name']}", timeout=2)

    @work(exclusive=True)
    async def _delete_project(self, project: dict) -> None:
        await self._project_manager.delete_project(project["id"])
        if self.state.current_project and self.state.current_project["id"] == project["id"]:
            self.state.current_project = None
            self._update_menu_bar()
        self.notify(f"已刪除專案：{project['name']}", timeout=2)
