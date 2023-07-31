from fastapi import APIRouter

from modules.db_connection.models.responses import DatabaseResponse
from modules.db_connection.service import DBConnectionService

router = APIRouter(prefix="/databases")


@router.get("/")
async def get_databases() -> list[DatabaseResponse]:
    return DBConnectionService().get_databases()
