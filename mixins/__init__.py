"""De-insight Mixins 包

包含應用的所有 Mixin 類，用來組織 DeInsightApp 的功能：
- ChatMixin: 對話、串流、互動提問
- MemoryMixin: 記憶管理、抽取、持久化
- RAGMixin: 知識圖譜、檢索、文件導入
- ProjectMixin: 專案切換、管理
- UIMixin: 菜單列、狀態列、模態框、通知

各 Mixin 對 App 的依賴介面定義於 protocol.py（DeInsightProtocol）。
新增 Mixin 方法時，如果涉及跨 Mixin 呼叫，應先檢查並更新 Protocol。
"""
