import collections


__all__ = ['Struct', 'ListOf', 'PairOf', 'Tensor']


no_default = object()


class Field():
    def __init__(self, checkers, default=no_default, doc=None):
        try:
            checkers = tuple(checkers)
        except TypeError:
            checkers = (checkers,)
        self.checkers = checkers
        self.default = default
        self.__doc__ = doc
        self.name = None

    def build_property(self, name):
        doc = self.__dict__.get('__doc__')
        result = Field(self.checkers, self.default, doc)
        result.name = name
        return result

    def __get__(self, obj, objtype):
        if obj is None:
            return self
        if self.name not in obj.__dict__:
            #
            if self.default is no_default:
                raise AttributeError('attribute %r of %r object is not set'
                                     % (self.name, type(self).__name__))
            #
            if not callable(self.default):
                default = self.default
            else:
                default = self.default()
            setattr(obj, self.name, default)
        return obj.__dict__[self.name]

    def __set__(self, obj, value):
        if not any(self._check(checker, value) for checker in self.checkers):
            raise ValueError(
                'invalid value for attribute %r: %r' % (self.name, value)
            )
        obj.__dict__[self.name] = value

    @staticmethod
    def _check(checker, value):
        if checker is None:
            return value is None
        elif isinstance(checker, type):
            return isinstance(value, checker)
        return checker(value)

    def __str__(self):
        names = []
        for checker in self.checkers:
            try:
                name = checker.__name__
            except Exception:
                name = str(checker)
            names.append(name)
        if len(names) == 1:
            return '%s(%s)' % (type(self).__name__, names[0])
        return '%s([%s])' % (type(self).__name__, ', '.join(names))

    @property
    def __doc__(self):
        doc = self.__dict__['__doc__']
        return '%s\n%s' % (self, doc) if doc else str(self)

    @__doc__.setter
    def __doc__(self, value):
        self.__dict__['__doc__'] = value


class StructMeta(type):
    Field = Field

    def __new__(cls, name, bases, namespace):
        namespace = dict(namespace)
        for key, value in namespace.items():
            if isinstance(value, Field):
                namespace[key] = value.build_property(key)
        return type.__new__(cls, name, bases, namespace)


class Struct(metaclass=StructMeta):
    def __new__(cls, **kwargs):
        obj = super().__new__(cls)
        for key, value in cls.__dict__.items():
            if isinstance(value, Field) and key in kwargs:
                setattr(obj, key, kwargs.pop(key))
        if kwargs:
            raise TypeError(
                "unexpected keyword arguments %s at creation of %s object"
                % (list(kwargs), cls.__name__ )
            )
        return obj


class ListOfMeta(type):
    def __new__(cls, name, bases, namespace):
        return type.__new__(cls, name, bases, namespace)

    def __instancecheck__(cls, obj):
        return (
            isinstance(type(obj), type(cls))
            and issubclass(obj._itemtype, cls._itemtype)
        )

    def __getitem__(cls, itemtype):
        name = '%s[%s]' % (cls.__name__, itemtype.__name__)
        return type(cls)(name, (cls,), dict(_itemtype=itemtype))

    def __iter__(cls):
        raise TypeError('%r object is not iterable' % cls.__name__)


class ListOf(list, metaclass=ListOfMeta):
    """ ListOf[itemtype]
    """
    def _check(self, value):
        if not isinstance(value, self._itemtype):
            raise TypeError('invalid type for value %r, should be %s'
                            % (value, self._itemtype.__name__))

    def __init__(self, iterable=None):
        if iterable is not None:
            self.extend(iterable)

    def __setitem__(self, idx, value):
        self._check(value)
        super().__setitem__(idx, value)

    def insert(self, idx, value):
        self._check(value)
        super().insert(idx, value)

    def append(self, value):
        self._check(value)
        super().append(value)

    def extend(self, values):
        values = list(values)
        for value in values:
            self._check(value)
        super().extend(values)


class PairMeta(type):
    def __instancecheck__(cls, obj):
        return (
            isinstance(obj, collections.Sequence)
            and len(obj) == 2
            and all(isinstance(val, cls._itemtype) for val in obj)
        )

    def __getitem__(cls, itemtype):
        name = '%s[%s]' % (cls.__name__, itemtype.__name__)
        return type(cls)(name, (cls,), dict(_itemtype=itemtype))

    def __iter__(cls):
        raise TypeError('%r object is not iterable' % cls.__name__)


class PairOf(metaclass=PairMeta):
    """ PairOf[itemtype]
    """
    def __new__(cls, *args, **kwargs):
        raise TypeError("Can't instantiate class %s" % cls.__name__)


def get_len(obj):
    " return len(obj) if possible, None otherwise"
    try:
        return len(obj)
    except TypeError:
        return None


class TensorMeta(type):
    def __instancecheck__(cls, obj):
        if not cls._shape:
            return isinstance(obj, cls._itemtype)
        dim, *subshape = cls._shape
        if get_len(obj) != dim:
            return False
        params = cls._itemtype, *subshape
        subcls = cls._parent[params]
        return all(isinstance(subobj, subcls) for subobj in obj)

    def __getitem__(cls, parameters):
        try:
            itemtype, *shape = parameters
            shape = tuple(shape)
        except TypeError:
            itemtype, shape = parameters, ()
        if not isinstance(itemtype, type):
            raise TypeError("first parameter must be a type")
        if not all(isinstance(d, int) and d>0 for d in shape):
            raise ValueError("dimensions must be positive int")
        name = '%s[%s, %s]' % (
            cls.__name__, itemtype.__name__, ', '.join(map(str, shape))
        )
        namespace = dict(_itemtype=itemtype, _shape=shape, _parent=cls)
        return type(cls)(name, (cls,), namespace)

    def __iter__(cls):
        raise TypeError('%r object is not iterable' % cls.__name__)


class Tensor(metaclass=TensorMeta):
    """ Tensor[itemtype, dim1, dim2, ...]
    """
    def __new__(cls, *args, **kwargs):
        raise TypeError("Can't instantiate class %s" % cls.__name__)
