"""Convert pre-emission automatic schedule admission errors into refusals.

This helper grants no execution authority. An admitted result must retain and
revalidate its original source, storage, numerical and completion obligations.
The caller constructs the checked incumbent after an automatic attempt refuses;
explicitly requested schedules retain their own error behavior.
"""


def attempt_checked_schedule(planner,*args):
    """Return (checked result, refusal, exception receipt) without emission."""
    try:
        result,refusal=planner(*args)
        if result is not None:
            result.revalidate()
        return result,refusal,None
    except (ValueError,TimeoutError) as error:
        receipt={'stage':'pre_emission_schedule_admission',
                 'exception':type(error).__name__,'reason':str(error),
                 'source_obligations':'unchanged; refused attempt is not emitted'}
        return None,type(error).__name__+': '+str(error),receipt
