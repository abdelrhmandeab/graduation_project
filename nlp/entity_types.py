from enum import Enum


class EntityType(str, Enum):
    APP = "app"
    SYSTEM_FEATURE = "system_feature"
    PATH = "path"
    NUMBER = "number"
    DURATION = "duration"
    EMAIL = "email"
    PERSON = "person"
    DATE = "date"


# Helper mapping for quick use
ENTITY_TYPE_MAP = {e.value: e for e in EntityType}
