import json
import logging
import os
from abc import ABC, abstractmethod
from typing import Any, Optional

from config import get_key
import redis
from session import SessionManager

logger = logging.getLogger(__name__)


class ReportStorage(ABC):
    """
    Abstract interface for report storage to maintain clean architecture.
    """
    @abstractmethod
    def save_report(self, session_id: str, report: dict[str, Any]) -> None:
        pass

    @abstractmethod
    def get_report(self, session_id: str) -> Optional[dict[str, Any]]:
        pass


    @abstractmethod
    def delete_report(self, session_id: str) -> None:
        pass


class HybridReportStorage(ReportStorage):
    """
    Stores reports in both Redis (for persistence) and RAM (SessionManager).
    If Redis fails, it falls back gracefully to RAM.
    When fetching, it tries Redis first, then RAM.
    """
    def __init__(self, session_manager: SessionManager):
        self.session_manager = session_manager
        self.expire_seconds = 86400  # 24 hours
        
        # Determine the redis URL from the config.
        # Fallback to standard REDIS_URL if reddis_url is not found.
        url = get_key("reddis_url") or get_key("REDIS_URL")
        
        self.redis_client = None
        if url:
            try:
                # Upstash REST URLs start with https://, but redis-py needs redis:// or rediss://
                # We attempt to connect. If it fails, redis_client remains None.
                self.redis_client = redis.from_url(url, decode_responses=True)
                logger.info(f"Initialized Redis client with URL starting with: {url[:10]}...")
            except Exception as e:
                logger.warning(f"Could not initialize Redis client (invalid URL format?): {e}")
        else:
            logger.info("No Redis URL found in environment variables. Defaulting to RAM-only storage.")

    def save_report(self, session_id: str, report: dict[str, Any]) -> None:
        """
        Saves the report to both Redis and the in-memory session manager.
        """
        # 1. Always save in RAM (SessionManager)
        try:
            self.session_manager.complete_session(session_id, report=report)
        except Exception as e:
            logger.error(f"Failed to save report to RAM for session {session_id}: {e}")

        # 2. Try to save in Redis
        if self.redis_client:
            redis_key = f"interview_report:{session_id}"
            try:
                report_json = json.dumps(report)
                self.redis_client.setex(
                    name=redis_key,
                    time=self.expire_seconds,
                    value=report_json
                )
                logger.info(f"Report for session {session_id} saved to Redis successfully.")
            except Exception as e:
                logger.warning(f"Failed to save report to Redis for session {session_id}: {e}. (Saved to RAM only)")

    def get_report(self, session_id: str) -> Optional[dict[str, Any]]:
        """
        Fetches the report. Checks Redis first. If not found or if Redis fails, checks RAM.
        """
        # 1. Try fetching from Redis
        if self.redis_client:
            redis_key = f"interview_report:{session_id}"
            try:
                report_json = self.redis_client.get(redis_key)
                if report_json:
                    logger.info(f"Successfully fetched report from Redis for session {session_id}.")
                    return json.loads(report_json)
            except Exception as e:
                logger.warning(f"Failed to fetch report from Redis for session {session_id}: {e}. Falling back to RAM.")

        # 2. Fallback to RAM (SessionManager)
        try:
            report = self.session_manager.get_session_report(session_id)
            if report:
                logger.info(f"Successfully fetched report from RAM for session {session_id}.")
                return report
        except Exception as e:
            logger.error(f"Failed to fetch report from RAM for session {session_id}: {e}")

        logger.info(f"Report not found in Redis or RAM for session {session_id}.")
        return None

    def delete_report(self, session_id: str) -> None:
        """
        Deletes the report from both Redis and RAM.
        """
        if self.redis_client:
            redis_key = f"interview_report:{session_id}"
            try:
                self.redis_client.delete(redis_key)
                logger.info(f"Deleted report from Redis for session {session_id}.")
            except Exception as e:
                logger.error(f"Failed to delete report from Redis for session {session_id}: {e}")
        
        try:
            # Note: session_manager.delete_session deletes the whole session, not just the report
            self.session_manager.delete_session(session_id)
        except Exception as e:
            logger.error(f"Failed to delete session from RAM for session {session_id}: {e}")
