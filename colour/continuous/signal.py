#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Signal
======

Defines the class implementing support for continuous signal:

-   :class:`Signal`
"""

from __future__ import division, unicode_literals

import numpy as np
from collections import Iterator, Mapping, OrderedDict, Sequence
from operator import add, mul, pow, sub, iadd, imul, ipow, isub

# Python 3 compatibility.
try:
    from operator import div, idiv
except ImportError:
    from operator import truediv, itruediv

    div = truediv
    idiv = itruediv

from colour.algebra import Extrapolator, LinearInterpolator
from colour.continuous import AbstractContinuousFunction
from colour.utilities import (as_numeric, closest, fill_nan,
                              is_pandas_installed, ndarray_write, tsplit,
                              tstack, warning)

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2017 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = ['Signal']


class Signal(AbstractContinuousFunction):
    """
    Defines the base class for continuous signal.

    The class implements the :meth:`Signal.function` method so that evaluating
    the function for any independent domain :math:`x \in \mathbb{R}` variable
    returns a corresponding range :math:`y \in \mathbb{R}` variable.
    It adopts an interpolating function encapsulated inside an extrapolating
    function. The resulting function independent domain, stored as discrete
    values in the :attr:`Signal.domain` attribute corresponds with the function
    dependent and already known range stored in the :attr:`Signal.range`
    attribute.

    Parameters
    ----------
    data : Series or Signal or array_like or dict_like, optional
        Data to be stored in the continuous signal.
    domain : array_like, optional
        Values to initialise the :attr:`Signal.domain` attribute with.
        If both `data` and `domain` arguments are defined, the latter with be
        used to initialise the :attr:`Signal.domain` attribute.

    Other Parameters
    ----------------
    name : unicode, optional
        Continuous signal name.
    interpolator : object, optional
        Interpolator class type to use as interpolating function.
    interpolator_args : dict_like, optional
        Arguments to use when instantiating the interpolating function.
    extrapolator : object, optional
        Extrapolator class type to use as extrapolating function.
    extrapolator_args : dict_like, optional
        Arguments to use when instantiating extrapolating function.

    Attributes
    ----------
    domain
    range
    interpolator
    interpolator_args
    extrapolator
    extrapolator_args
    function

    Methods
    -------
    __str__
    __repr__
    __getitem__
    __setitem__
    __contains__
    __eq__
    __neq__
    arithmetical_operation
    signal_unpack_data
    fill_nan
    uncertainty
    """

    def __init__(self, data=None, domain=None, **kwargs):
        super(Signal, self).__init__(kwargs.get('name'))

        self._domain = None
        self._range = None
        self._interpolator = LinearInterpolator
        self._interpolator_args = {}
        self._extrapolator = Extrapolator
        self._extrapolator_args = {
            'method': 'Constant',
            'left': np.nan,
            'right': np.nan
        }

        self.domain, self.range = self.signal_unpack_data(data, domain)

        self.interpolator = kwargs.get('interpolator')
        self.interpolator_args = kwargs.get('interpolator_args')
        self.extrapolator = kwargs.get('extrapolator')
        self.extrapolator_args = kwargs.get('extrapolator_args')

        self._create_function()

    @property
    def domain(self):
        """
        Getter and setter property for the continuous signal independent
        domain :math:`x` variable.

        Parameters
        ----------
        value : array_like
            Value to set the continuous signal independent domain
            :math:`x` variable with.

        Returns
        -------
        ndarray
            Continuous signal independent domain :math:`x` variable.
        """

        return self._domain

    @domain.setter
    def domain(self, value):
        """
        Setter for the **self.domain** property.
        """

        if value is not None:
            if not np.all(np.isfinite(value)):
                warning('"domain" variable is not finite, '
                        'unpredictable results may occur!\n{0}'.format(value))

            # TODO: `self.domain` is a copy of `value` to avoid side effects,
            # Is it a smart way to avoid them?
            value = np.copy(np.asarray(value))

            if self._range is not None:
                assert value.size == self._range.size, (
                    '"domain" and "range" variables must have same size!')

            value.setflags(write=False)
            self._domain = value
            self._create_function()

    @property
    def range(self):
        """
        Getter and setter property for the continuous signal corresponding
        range :math:`y` variable.

        Parameters
        ----------
        value : array_like
            Value to set the continuous signal corresponding range :math:`y`
            variable with.

        Returns
        -------
        ndarray
            Continuous signal corresponding range :math:`y` variable.
        """

        return self._range

    @range.setter
    def range(self, value):
        """
        Setter for the **self.range** property.
        """

        if value is not None:
            if not np.all(np.isfinite(value)):
                warning('"range" variable is not finite, '
                        'unpredictable results may occur!\n{0}'.format(value))

            # TODO: `self.range` is a copy of `value` to avoid side effects,
            # Is it a smart way to avoid them?
            value = np.copy(np.asarray(value))

            if self._domain is not None:
                assert value.size == self._domain.size, (
                    '"domain" and "range" variables must have same size!')

            value.setflags(write=False)
            self._range = value
            self._create_function()

    @property
    def interpolator(self):
        """
        Getter and setter property for the continuous signal interpolator type.

        Parameters
        ----------
        value : type
            Value to set the continuous signal interpolator type
            with.

        Returns
        -------
        type
            continuous signal interpolator type.
        """

        return self._interpolator

    @interpolator.setter
    def interpolator(self, value):
        """
        Setter for the **self.interpolator** property.
        """

        if value is not None:
            # TODO: Check for interpolator capabilities.
            self._interpolator = value
            self._create_function()

    @property
    def interpolator_args(self):
        """
        Getter and setter property for the continuous signal interpolator
        instantiation time arguments.

        Parameters
        ----------
        value : dict
            Value to set the continuous signal interpolator instantiation
            time arguments to.

        Returns
        -------
        dict
            Continuous signal interpolator instantiation time
            arguments.
        """

        return self._interpolator_args

    @interpolator_args.setter
    def interpolator_args(self, value):
        """
        Setter for the **self.interpolator_args** property.
        """

        if value is not None:
            assert type(value) in (dict, OrderedDict), ((
                '"{0}" attribute: "{1}" type is not '
                '"dict" or "OrderedDict"!').format('interpolator_args', value))

            self._interpolator_args = value
            self._create_function()

    @property
    def extrapolator(self):
        """
        Getter and setter property for the continuous signal extrapolator type.

        Parameters
        ----------
        value : type
            Value to set the continuous signal extrapolator type
            with.

        Returns
        -------
        type
            Continuous signal extrapolator type.
        """

        return self._extrapolator

    @extrapolator.setter
    def extrapolator(self, value):
        """
        Setter for the **self.extrapolator** property.
        """

        if value is not None:
            # TODO: Check for extrapolator capabilities.
            self._extrapolator = value
            self._create_function()

    @property
    def extrapolator_args(self):
        """
        Getter and setter property for the continuous signal extrapolator
        instantiation time arguments.

        Parameters
        ----------
        value : dict
            Value to set the continuous signal extrapolator instantiation
            time arguments to.

        Returns
        -------
        dict
            Continuous signal extrapolator instantiation time
            arguments.
        """

        return self._extrapolator_args

    @extrapolator_args.setter
    def extrapolator_args(self, value):
        """
        Setter for the **self.extrapolator_args** property.
        """

        if value is not None:
            assert type(value) in (dict, OrderedDict), ((
                '"{0}" attribute: "{1}" type is not '
                '"dict" or "OrderedDict"!').format('extrapolator_args', value))

            self._extrapolator_args = value
            self._create_function()

    @property
    def function(self):
        """
        Getter and setter property for the continuous signal callable.

        Parameters
        ----------
        value : object
            Attribute value.

        Returns
        -------
        callable
            Continuous signal callable.

        Notes
        -----
        -   This property is read only.
        """

        return self._function

    @function.setter
    def function(self, value):
        """
        Setter for the **self.function** property.
        """

        raise AttributeError(
            '"{0}" attribute is read only!'.format('function'))

    def __str__(self):
        """
        Returns a formatted string representation of the continuous signal.

        Returns
        -------
        unicode
            Formatted string representation.
        """

        try:
            return str(tstack((self.domain, self.range)))
        except TypeError:
            return super(Signal, self).__str__()

    def __repr__(self):
        """
        Returns an evaluable string representation of the continuous signal.

        Returns
        -------
        unicode
            Evaluable string representation.
        """

        try:
            representation = repr(tstack((self.domain, self.range)))
            representation = representation.replace('array',
                                                    self.__class__.__name__)
            representation = representation.replace('       [', '{0}['.format(
                ' ' * (len(self.__class__.__name__) + 2)))
            representation = ('{0},\n'
                              '{1}interpolator={2},\n'
                              '{1}interpolator_args={3},\n'
                              '{1}extrapolator={4},\n'
                              '{1}extrapolator_args={5})').format(
                                  representation[:-1],
                                  ' ' * (len(self.__class__.__name__) + 1),
                                  self.interpolator.__name__,
                                  repr(self.interpolator_args),
                                  self.extrapolator.__name__,
                                  repr(self.extrapolator_args))

            return representation
        except TypeError:
            # TODO: Discuss what is the most suitable behaviour, either the
            # following or __str__ one.
            return '{0}()'.format(self.__class__.__name__)

    def __getitem__(self, x):
        """
        Returns the corresponding range :math:`y` variable for independent
        domain :math:`x` variable.

        Parameters
        ----------
        x : numeric, array_like or slice
            Independent domain :math:`x` variable.

        Returns
        -------
        numeric or ndarray
            math:`y` range value.
        """

        if type(x) is slice:
            return self._range[x]
        else:
            return self._function(x)

    def __setitem__(self, x, y):
        """
        Sets the corresponding range :math:`y` variable for independent domain
        :math:`x` variable.

        Parameters
        ----------
        x : numeric, array_like or slice
            Independent domain :math:`x` variable.
        y : numeric or ndarray
            Corresponding range :math:`y` variable.
        """

        if type(x) is slice:
            with ndarray_write(self._range):
                self._range[x] = y
        else:
            with ndarray_write(self._domain), ndarray_write(self._range):
                x = np.atleast_1d(x)
                y = np.resize(y, x.shape)

                # Matching domain, replacing existing `self.range`.
                mask = np.in1d(x, self._domain)
                x_m = x[mask]
                indexes = np.searchsorted(self._domain, x_m)
                self._range[indexes] = y[mask]

                # Non matching domain, inserting into existing `self.domain`
                # and `self.range`.
                x_nm = x[~mask]
                indexes = np.searchsorted(self._domain, x_nm)
                if indexes.size != 0:
                    self._domain = np.insert(self._domain, indexes, x_nm)
                    self._range = np.insert(self._range, indexes, y[~mask])

        self._create_function()

    def __contains__(self, x):
        """
        Returns whether the continuous signal contains given independent domain
        :math:`x` variable.

        Parameters
        ----------
        x : numeric, array_like or slice
            Independent domain :math:`x` variable.

        Returns
        -------
        bool
            Is :math:`x` domain value contained.
        """

        return np.all(
            np.where(
                np.logical_and(x >= np.min(self._domain), x <=
                               np.max(self._domain)), True, False))

    def __eq__(self, other):
        """
        Returns whether the continuous signal is equal to given other object.

        Parameters
        ----------
        other : object
            Object to test whether it is equal to continuous signal.

        Returns
        -------
        bool
            Is given object equal to the continuous signal.
        """

        if isinstance(other, Signal):
            if all([
                    np.array_equal(self._domain, other.domain),
                    np.array_equal(self._range, other.range),
                    self._interpolator is other.interpolator,
                    self._interpolator_args == other.interpolator_args,
                    self._extrapolator is other.extrapolator,
                    self._extrapolator_args == other.extrapolator_args
            ]):
                return True

        return False

    def __neq__(self, other):
        """
        Returns whether the continuous signal is not equal to given other
        object.

        Parameters
        ----------
        other : object
            Object to test whether it is not equal to the continuous signal.

        Returns
        -------
        bool
            Is given object not equal to the continuous signal.
        """

        return not (self == other)

    def _create_function(self):
        """
        Creates the continuous signal underlying function.
        """

        if self._domain is not None and self._range is not None:
            with ndarray_write(self._domain), ndarray_write(self._range):
                # TODO: Providing a writeable copy of both `self.domain` and `
                # self.range` to the interpolator to avoid issue regarding
                # `MemoryView` being read-only.
                # https://mail.python.org/pipermail/cython-devel/2013-February/003384.html
                self._function = self._extrapolator(
                    self._interpolator(
                        np.copy(self._domain),
                        np.copy(self._range), **self._interpolator_args),
                    **self._extrapolator_args)
        else:

            def _undefined_function(*args, **kwargs):
                raise RuntimeError(
                    'Underlying signal interpolator function does not exists, '
                    'please ensure you defined both '
                    '"domain" and "range" variables!')

            self._function = _undefined_function

    def _fill_domain_nan(self, method='Interpolation', default=0):
        """
        Fill NaNs in independent domain :math:`x` variable using given method.

        Parameters
        ----------
        method : unicode, optional
            **{'Interpolation', 'Constant'}**,
            *Interpolation* method linearly interpolates through the NaNs,
            *Constant* method replaces NaNs with `default`.
        default : numeric, optional
            Value to use with the *Constant* method.

        Returns
        -------
        Signal
            NaNs filled continuous signal independent domain :math:`x`
            variable.
        """

        with ndarray_write(self._domain):
            self._domain = fill_nan(self._domain, method, default)
            self._create_function()

    def _fill_range_nan(self, method='Interpolation', default=0):
        """
        Fill NaNs in corresponding range :math:`y` variable using given method.

        Parameters
        ----------
        method : unicode, optional
            **{'Interpolation', 'Constant'}**,
            *Interpolation* method linearly interpolates through the NaNs,
            *Constant* method replaces NaNs with `default`.
        default : numeric, optional
            Value to use with the *Constant* method.

        Returns
        -------
        Signal
            NaNs filled continuous signal i corresponding range :math:`y`
            variable.
        """

        with ndarray_write(self._range):
            self._range = fill_nan(self._range, method, default)
            self._create_function()

    def arithmetical_operation(self, a, operator, in_place=False):
        """
        Performs given arithmetical operation with :math:`a` operand, the
        operation can be either performed on a copy or in-place.

        Parameters
        ----------
        a : numeric or ndarray or Signal
            Operand.
        operator : object
            Operation to perform.
        in_place : bool, optional
            Operation happens in place.

        Returns
        -------
        Signal
            Continuous signal.
        """

        operator, ioperator = {
            '+': (add, iadd),
            '-': (sub, isub),
            '*': (mul, imul),
            '/': (div, idiv),
            '**': (pow, ipow)
        }[operator]

        if in_place:
            if isinstance(a, Signal):
                with ndarray_write(self._domain), ndarray_write(self._range):
                    self[self._domain] = operator(self._range, a[self._domain])

                    exclusive_or = np.setxor1d(self._domain, a.domain)
                    self[exclusive_or] = np.full(exclusive_or.shape, np.nan)
            else:
                with ndarray_write(self._range):
                    self.range = ioperator(self.range, a)

            return self
        else:
            copy = ioperator(self.copy(), a)

            return copy

    @staticmethod
    def signal_unpack_data(data=None, domain=None):
        """
        Unpack given data for continuous signal instantiation.

        Parameters
        ----------
        data : Series or Signal or array_like or dict_like, optional
            Data to unpack for continuous signal instantiation.
        domain : array_like, optional
            Values to initialise the :attr:`Signal.domain` attribute with.
            If both `data` and `domain` arguments are defined, the latter with
            be used to initialise the :attr:`Signal.domain` attribute.

        Returns
        -------
        tuple
            Independent domain :math:`x` variable and corresponding range
            :math:`y` variable unpacked for continuous signal instantiation.
        """

        domain_upk, range_upk = None, None
        if isinstance(data, Signal):
            domain_upk = data.domain
            range_upk = data.range
        elif (issubclass(type(data), Sequence) or
              isinstance(data, (tuple, list, np.ndarray, Iterator))):
            data = tsplit(list(data) if isinstance(data, Iterator) else data)
            assert data.ndim in (1, 2), (
                'User "data" must be a 1d or 2d array-like variable!')
            if data.ndim == 1:
                domain_upk, range_upk = np.arange(0, data.size), data
            else:
                domain_upk, range_upk = data
        elif (issubclass(type(data), Mapping) or
              isinstance(data, (dict, OrderedDict))):
            domain_upk, range_upk = tsplit(sorted(data.items()))
        elif is_pandas_installed():
            from pandas import Series

            if isinstance(data, Series):
                domain_upk = data.index.values
                range_upk = data.values

        if domain is not None and range_upk is not None:
            assert len(domain) == len(range_upk), (
                'User "domain" is not compatible with unpacked range!')
            domain_upk = domain

        return domain_upk, range_upk

    def fill_nan(self, method='Interpolation', default=0):
        """
        Fill NaNs in independent domain :math:`x` variable and corresponding
        range :math:`y` variable using given method.

        Parameters
        ----------
        method : unicode, optional
            **{'Interpolation', 'Constant'}**,
            *Interpolation* method linearly interpolates through the NaNs,
            *Constant* method replaces NaNs with `default`.
        default : numeric, optional
            Value to use with the *Constant* method.

        Returns
        -------
        Signal
            NaNs filled continuous signal.
        """

        self._fill_domain_nan(method, default)
        self._fill_range_nan(method, default)

        return self

    def uncertainty(self, a):
        """
        Returns the uncertainty between independent domain :math:`x` variable
        and given :math:`a` variable.

        Parameters
        ----------
        a : numeric or array_like
            :math:`a` variable to compute the uncertainty with independent
            domain :math:`x` variable.

        Returns
        -------
        numeric or array_like
            Uncertainty between independent domain :math:`x` variable and given
            :math:`a` variable.
        """

        n = closest(self._domain, a)

        return as_numeric(np.abs(a - n))
