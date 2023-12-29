class AxisAlignedBoundingBox:
    def __init__(self, intervals):
        self._intervals = intervals

    def contains_obs_given_dims(self, obs, dim_idxs):
        for dim_idx in dim_idxs:
            if not (self._intervals[dim_idx]).contains_val(obs[dim_idx]):
                return False
        return True

    def __getitem__(self, dim_idx):
        return self._intervals[dim_idx]

    def __iter__(self):
        return iter(self._intervals)
