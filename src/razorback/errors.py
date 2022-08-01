""" Exception classes specific to the package
"""


class NonConvergence(Exception):
    def __str__(self):
        return ' '.join(map(str, self.args))
