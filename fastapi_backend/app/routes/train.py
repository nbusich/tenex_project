from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path
from typing import Any
import asyncio
import json
import traceback

from app.database import User
from app.users import current_active_user
from app.anomaly.train.train.train_autoencoder import train_autoencoder
from app.anomaly.train.train.train_transformer import train_transformer
from app.anomaly.train.train.train_rf import train_rf

from fastapi import APIRouter, Depends, HTTPException, Query, status

from app.schemas import (
    TrainResponse,
    TrainConfigRead,
    TrainMetricsRead
)
from app.models import (
    TrainConfig, 
    TrainMetrics
    )

from app.database import User, get_async_session
from sqlalchemy.ext.asyncio import AsyncSession

router = APIRouter(tags=["train"])
LOG = logging.getLogger(__name__)

@router.post("", response_model=TrainResponse)
async def train_model(
    user_input: TrainConfigRead = None,
    db: AsyncSession = Depends(get_async_session),
    user: User = Depends(current_active_user),
):
    # Pydantic handles all type checking, we just have to handle content checking and whether the pydantic thing exists at all
    if user_input is None:
        return HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No user input detected"
        )

    input_dict = user_input.model_dump()
    model_name = input_dict.get("model_name", 'autoencoder')
    lr = input_dict.get("lr", 1e-3)
    epochs = input_dict.get("epochs", 10)
    batch_size = input_dict.get("batch_size", 128)

    try:
        if model_name == "transformer":
            metrics = await asyncio.to_thread(train_transformer,
            batch_size=batch_size, 
            lr=lr, 
            epochs=epochs)

        elif model_name == "autoencoder":
            metrics = await asyncio.to_thread(train_autoencoder,
            batch_size=batch_size, 
            lr=lr, 
            epochs=epochs)

        else:
            metrics = await asyncio.to_thread(train_rf)
    
    except Exception as e:
        error_stack = traceback.format_exc()
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Training Failed: {error_stack}"
        )
    
    s = "Success"

    if type(metrics) != dict:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Metrics did not output a dictionary"
        )

    config = TrainConfig(
        user_id=user.id,
        model_name = model_name,
        lr = lr,
        epochs = epochs,
        batch_size = batch_size,
        status = s
    )

    try:
        try:
            db.add(config)
            await db.flush() # populate config id
        except Exception as e:
            error_stack = traceback.format_exc()
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="First add statement"
            )

        metrics = TrainMetrics(
            run_id=config.id, # use populated config id
            f1=metrics.get('f1', 0),
            precision=metrics.get('precision', 0),
            recall=metrics.get('recall', 0),
            n=metrics.get('n', 0),
            n_positive=metrics.get('n_positive', 0)
        )

        db.add(metrics)
        await db.commit()
    except Exception as e:
        error_stack = traceback.format_exc()
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Database failed: {error_stack}"
        )

    try:
        cfg = TrainConfigRead.model_validate(config)
        mtr = TrainMetricsRead.model_validate(metrics)
    except Exception as e:
        error_stack = traceback.format_exc()
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Training Failed: {error_stack}"
        )
    
    return TrainResponse(
        config = cfg,
        metrics = mtr,
        status = s
    )