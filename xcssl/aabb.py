class AxisAlignedBoundingBox:
    def __init__(self, intervals):
        self._intervals = tuple(intervals)

    def contains_obs_all_dims(self, obs):
        for (interval, obs_compt) in zip(self._intervals, obs):
            if not interval.contains_val(obs_compt):
                return False
        return True

    def contains_obs_given_dims(self, obs, dim_idxs):
        for dim_idx in dim_idxs:
            if not (self._intervals[dim_idx]).contains_val(obs[dim_idx]):
                return False
        return True

    def __getitem__(self, dim_idx):
        return self._intervals[dim_idx]

    def __iter__(self):
        return iter(self._intervals)