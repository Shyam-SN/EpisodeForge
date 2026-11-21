"""Cost model rate configuration.

NOTE: These rates are user-configurable reference rates and should be verified
against current cloud provider pricing before financial reporting.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class CostRates(BaseModel):
    # Compute: rate per core-hour ($/vCPU-hr)
    compute_core_hour_usd: float = Field(default=0.045, ge=0.0)

    # Storage: rate per GB-month ($/GB-mo)
    s3_standard_storage_gb_month_usd: float = Field(default=0.023, ge=0.0)

    # Egress: internet data transfer out per GB ($/GB)
    s3_egress_gb_usd: float = Field(default=0.090, ge=0.0)

    # API Requests: S3 PUT/POST per 1,000 requests ($/1k)
    s3_put_request_per_1k_usd: float = Field(default=0.005, ge=0.0)

    # S3 GET/LIST per 10,000 requests ($/10k)
    s3_get_request_per_10k_usd: float = Field(default=0.004, ge=0.0)
