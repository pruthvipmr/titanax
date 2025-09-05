"""JAX distributed initialization and device utilities.

This module provides utilities for initializing JAX in distributed environments
and enumerating available devices.
"""

import os
import logging
from typing import Optional, List, Dict, Any

import jax

from ..exceptions import DistributedError
from ..types import Device


logger = logging.getLogger(__name__)


def detect_distributed_env() -> Dict[str, Optional[str]]:
    """Detect distributed environment variables.
    
    Checks for common multi-host environment variables used by JAX.
    
    Returns:
        Dict containing detected environment variables and their values.
        Keys: coordinator_address, coordinator_port, process_count, process_id
    """
    env_vars = {
        'coordinator_address': os.environ.get('JAX_COORDINATOR_ADDRESS'),
        'coordinator_port': os.environ.get('JAX_COORDINATOR_PORT'),
        'process_count': os.environ.get('JAX_PROCESS_COUNT'),
        'process_id': os.environ.get('JAX_PROCESS_ID'),
    }
    
    # Also check for alternative environment variable names
    if not env_vars['coordinator_address']:
        env_vars['coordinator_address'] = os.environ.get('COORDINATOR_ADDRESS')
    if not env_vars['coordinator_port']:
        env_vars['coordinator_port'] = os.environ.get('COORDINATOR_PORT')
    if not env_vars['process_count']:
        env_vars['process_count'] = os.environ.get('WORLD_SIZE')
    if not env_vars['process_id']:
        env_vars['process_id'] = os.environ.get('RANK')
        
    return env_vars


def is_distributed_env() -> bool:
    """Check if running in a distributed environment.
    
    Returns:
        True if distributed environment variables are detected.
    """
    env_vars = detect_distributed_env()
    required_vars = ['coordinator_address', 'process_count', 'process_id']
    return all(env_vars[var] is not None for var in required_vars)


def initialize_distributed(
    coordinator_address: Optional[str] = None,
    coordinator_port: Optional[str] = None, 
    process_count: Optional[int] = None,
    process_id: Optional[int] = None,
    timeout_seconds: float = 300.0
) -> None:
    """Initialize JAX distributed runtime.
    
    Args:
        coordinator_address: IP address of the coordinator process
        coordinator_port: Port number for coordinator (default: 1234)
        process_count: Total number of processes in the job
        process_id: ID of this process (0-indexed)
        timeout_seconds: Timeout for initialization
        
    Raises:
        DistributedError: If initialization fails
    """
    # Use environment variables as fallback
    env_vars = detect_distributed_env()
    
    coordinator_address = coordinator_address or env_vars['coordinator_address']
    coordinator_port = coordinator_port or env_vars['coordinator_port'] or '1234'
    
    if process_count is None and env_vars['process_count']:
        process_count = int(env_vars['process_count'])
    if process_id is None and env_vars['process_id']:
        process_id = int(env_vars['process_id'])
    
    if not all([coordinator_address, process_count is not None, process_id is not None]):
        raise DistributedError(
            "Missing required distributed configuration",
            "Set JAX_COORDINATOR_ADDRESS, JAX_PROCESS_COUNT, and JAX_PROCESS_ID "
            "environment variables or pass them as arguments"
        )
    
    coordinator_address_with_port = f"{coordinator_address}:{coordinator_port}"
    
    # Skip if already initialized
    if jax.distributed.is_initialized():
        logger.info("JAX distributed already initialized")
        return
    
    try:
        logger.info(f"Initializing JAX distributed: coordinator={coordinator_address_with_port}, "
                   f"process_count={process_count}, process_id={process_id}")
        
        jax.distributed.initialize(
            coordinator_address=coordinator_address_with_port,
            num_processes=process_count,
            process_id=process_id,
            initialization_timeout=int(timeout_seconds)
        )
        
        logger.info(f"JAX distributed initialized successfully. "
                   f"Local devices: {len(jax.local_devices())}, "
                   f"Global devices: {jax.device_count()}")
        
    except Exception as e:
        raise DistributedError(
            f"Failed to initialize JAX distributed: {e}",
            "Check coordinator address/port, ensure all processes can reach coordinator, "
            "and verify process_count/process_id are correct"
        ) from e


def enumerate_devices(
    device_type: Optional[str] = None,
    local_only: bool = False
) -> List[Device]:
    """Enumerate available JAX devices.
    
    Args:
        device_type: Filter by device type ('gpu', 'tpu', 'cpu'). None for all.
        local_only: If True, return only local devices. If False, return all devices.
        
    Returns:
        List of JAX devices
    """
    if local_only:
        devices = jax.local_devices()
    else:
        devices = jax.devices()
    
    if device_type is not None:
        device_type = device_type.lower()
        devices = [d for d in devices if d.platform.lower() == device_type]
    
    return devices


def get_device_info() -> Dict[str, Any]:
    """Get information about available devices.
    
    Returns:
        Dictionary containing device counts and types
    """
    local_devices = jax.local_devices()
    global_devices = jax.devices()
    
    # Count devices by type
    local_by_type = {}
    global_by_type = {}
    
    for device in local_devices:
        device_type = device.platform.lower()
        local_by_type[device_type] = local_by_type.get(device_type, 0) + 1
        
    for device in global_devices:
        device_type = device.platform.lower()
        global_by_type[device_type] = global_by_type.get(device_type, 0) + 1
    
    return {
        'local_device_count': len(local_devices),
        'global_device_count': len(global_devices),
        'local_devices_by_type': local_by_type,
        'global_devices_by_type': global_by_type,
        'process_count': jax.process_count(),
        'process_index': jax.process_index(),
        'is_distributed': jax.process_count() > 1,
    }


def validate_device_availability(min_devices: int = 1) -> None:
    """Validate that sufficient devices are available.
    
    Args:
        min_devices: Minimum number of devices required
        
    Raises:
        DistributedError: If insufficient devices available
    """
    available_devices = len(jax.local_devices())
    
    if available_devices < min_devices:
        raise DistributedError(
            f"Insufficient devices: found {available_devices}, need {min_devices}",
            "Check device availability and ensure JAX can access the required devices"
        )


def auto_initialize() -> bool:
    """Automatically initialize distributed JAX if environment suggests it.
    
    Returns:
        True if distributed initialization was performed, False otherwise
    """
    if is_distributed_env():
        logger.info("Distributed environment detected, initializing JAX distributed")
        initialize_distributed()
        return True
    else:
        logger.info("Single-process environment detected")
        return False
