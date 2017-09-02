#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Multi Signal
============

Defines the class implementing support for multi-continuous signal:

-   :class:`MultiSignal`
"""

from __future__ import division, unicode_literals

import numpy as np
from collections import Iterator, Mapping, OrderedDict, Sequence

# Python 3 compatibility.
try:
    from operator import div, idiv
except ImportError:
    from operator import truediv, itruediv

    div = truediv
    idiv = itruediv

from colour.continuous import AbstractContinuousFunction, Signal
from colour.utilities import first_item, is_pandas_installed, tsplit, tstack

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2017 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = ['MultiSignal']


class MultiSignal(AbstractContinuousFunction):
    """
    Defines the base class for multi-continuous signal, a container for
    multiple :class:`Signal` class instances.

    Parameters
    ----------
    data : Series or Dataframe or Signal or MultiSignal or array_like or \
dict_like, optional
        Data to be stored in the multi-continuous signal.
    domain : array_like, optional
        Values to initialise the multiple :class:`Signal` class instances
        :attr:`Signal.domain` attribute with. If both `data` and `domain`
        arguments are defined, the latter with be used to initialise the
        :attr:`Signal.domain` attribute.
    labels : array_like, optional
        Names to use for the :class:`Signal` class instances.

    Other Parameters
    ----------------
    name : unicode, optional
        Multi-continuous signal name.
    interpolator : object, optional
        Interpolator class type to use as interpolating function for the
        :class:`Signal` class instances.
    interpolator_args : dict_like, optional
        Arguments to use when instantiating the interpolating function
        of the :class:`Signal` class instances.
    extrapolator : object, optional
        Extrapolator class type to use as extrapolating function for the
        :class:`Signal` class instances.
    extrapolator_args : dict_like, optional
        Arguments to use when instantiating extrapolating function
        of the :class:`Signal` class instances.

    Attributes
    ----------
    domain
    range
    interpolator
    interpolator_args
    extrapolator
    extrapolator_args
    function
    signals
    labels

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
    multi_signal_unpack_data
    fill_nan
    uncertainty
    """

    def __init__(self, data=None, domain=None, labels=None, **kwargs):
        super(MultiSignal, self).__init__(kwargs.get('name'))

        self._signals = self.multi_signal_unpack_data(data, domain, labels)

    @property
    def domain(self):
        """
        Getter and setter property for the :class:`Signal` class instances
        independent domain :math:`x` variable.

        Parameters
        ----------
        value : array_like
            Value to set the :class:`Signal` class instances independent domain
            :math:`x` variable with.

        Returns
        -------
        ndarray
            :class:`Signal` class instances independent domain :math:`x`
            variable.
        """

        if self._signals:
            return first_item(self._signals.values()).domain

    @domain.setter
    def domain(self, value):
        """
        Setter for the **self.domain** property.
        """

        if value is not None:
            for signal in self._signals.values():
                signal.domain = value

    @property
    def range(self):
        """
        Getter and setter property for the :class:`Signal` class instances
        corresponding range :math:`y` variable.

        Parameters
        ----------
        value : array_like
            Value to set the :class:`Signal` class instances corresponding
            range :math:`y` variable with.

        Returns
        -------
        ndarray
            :class:`Signal` class instances corresponding range :math:`y`
            variable.
        """

        if self._signals:
            return tstack([signal.range for signal in self._signals.values()])

    @range.setter
    def range(self, value):
        """
        Setter for the **self.range** property.
        """

        # TODO: Handle 2D array `value`.
        if value is not None:
            for signal in self._signals.values():
                signal.range = value

    @property
    def interpolator(self):
        """
        Getter and setter property for the :class:`Signal` class instances
        interpolator type.

        Parameters
        ----------
        value : type
            Value to set the :class:`Signal` class instances interpolator type
            with.

        Returns
        -------
        type
            :class:`Signal` class instances interpolator type.
        """

        if self._signals:
            return first_item(self._signals.values()).interpolator

    @interpolator.setter
    def interpolator(self, value):
        """
        Setter for the **self.interpolator** property.
        """

        if value is not None:
            for signal in self._signals.values():
                signal.interpolator = value

    @property
    def interpolator_args(self):
        """
        Getter and setter property for the :class:`Signal` class instances
        interpolator instantiation time arguments.

        Parameters
        ----------
        value : dict
            Value to set the :class:`Signal` class instances interpolator
            instantiation time arguments to.

        Returns
        -------
        dict
            :class:`Signal` class instances interpolator instantiation
            time arguments.
        """

        if self._signals:
            return first_item(self._signals.values()).interpolator_args

    @interpolator_args.setter
    def interpolator_args(self, value):
        """
        Setter for the **self.interpolator_args** property.
        """

        if value is not None:
            for signal in self._signals.values():
                signal.interpolator_args = value

    @property
    def extrapolator(self):
        """
        Getter and setter property for the :class:`Signal` class instances
        extrapolator type.

        Parameters
        ----------
        value : type
            Value to set the :class:`Signal` class instances extrapolator type
            with.

        Returns
        -------
        type
            :class:`Signal` class instances extrapolator type.
        """

        if self._signals:
            return first_item(self._signals.values()).extrapolator

    @extrapolator.setter
    def extrapolator(self, value):
        """
        Setter for the **self.extrapolator** property.
        """

        if value is not None:
            for signal in self._signals.values():
                signal.extrapolator = value

    @property
    def extrapolator_args(self):
        """
        Getter and setter property for the :class:`Signal` class instances
        extrapolator instantiation time arguments.

        Parameters
        ----------
        value : dict
            Value to set the :class:`Signal` class instances extrapolator
            instantiation time arguments to.

        Returns
        -------
        dict
            :class:`Signal` class instances extrapolator instantiation
            time arguments.
        """

        if self._signals:
            return first_item(self._signals.values()).extrapolator_args

    @extrapolator_args.setter
    def extrapolator_args(self, value):
        """
        Setter for the **self.extrapolator_args** property.
        """

        if value is not None:
            for signal in self._signals.values():
                signal.extrapolator_args = value

    @property
    def function(self):
        """
        Getter and setter property for the :class:`Signal` class instances
        callable.

        Parameters
        ----------
        value : object
            Attribute value.

        Returns
        -------
        callable
            :class:`Signal` class instances callable.

        Notes
        -----
        -   This property is read only.
        """

        if self._signals:
            return first_item(self._signals.values()).function

    @function.setter
    def function(self, value):
        """
        Setter for the **self.function** property.
        """

        raise AttributeError(
            '"{0}" attribute is read only!'.format('function'))

    @property
    def signals(self):
        """
        Getter and setter property for the :class:`Signal` class instances.

        Parameters
        ----------
        value : Series or Dataframe or Signal or MultiSignal or array_like or \
dict_like
            Attribute value.

        Returns
        -------
        OrderedDict
            :class:`Signal` class instances.
        """

        return self._signals

    @signals.setter
    def signals(self, value):
        """
        Setter for the **self.signals** property.
        """

        if value is not None:
            self._signals = self.multi_signal_unpack_data(value)

    @property
    def labels(self):
        """
        Getter and setter property for the :class:`Signal` class instances
        name.

        Parameters
        ----------
        value : array_like
            Value to set the :class:`Signal` class instances name.

        Returns
        -------
        dict
            :class:`Signal` class instance name.
        """

        if self._signals:
            return list(self._signals.keys())

    @labels.setter
    def labels(self, value):
        """
        Setter for the **self.labels** property.
        """

        if value is not None:
            assert len(value) == len(self._signals), (
                '"labels" length does not match "signals" length!')
            self._signals = OrderedDict(
                [(value[i], signal)
                 for i, (_key, signal) in enumerate(self._signals.items())])

    def __str__(self):
        """
        Returns a formatted string representation of the multi-continuous
        signal.

        Returns
        -------
        unicode
            Formatted string representation.
        """

        try:
            return str(np.hstack((self.domain[:, np.newaxis], self.range)))
        except TypeError:
            return super(MultiSignal, self).__str__()

    def __repr__(self):
        """
        Returns an evaluable string representation of the multi-continuous
        signal.

        Returns
        -------
        unicode
            Evaluable string representation.
        """

        try:
            representation = repr(
                np.hstack((self.domain[:, np.newaxis], self.range)))
            representation = representation.replace('array',
                                                    self.__class__.__name__)
            representation = representation.replace('       [', '{0}['.format(
                ' ' * (len(self.__class__.__name__) + 2)))
            representation = ('{0},\n'
                              '{1}labels={2},\n'
                              '{1}interpolator={3},\n'
                              '{1}interpolator_args={4},\n'
                              '{1}extrapolator={5},\n'
                              '{1}extrapolator_args={6})').format(
                                  representation[:-1],
                                  ' ' * (len(self.__class__.__name__) + 1),
                                  repr(self.labels), self.interpolator.__name__
                                  if self.interpolator is not None else
                                  self.interpolator,
                                  repr(self.interpolator_args),
                                  self.extrapolator.__name__
                                  if self.extrapolator is not None else
                                  self.extrapolator,
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

        if self._signals:
            return tstack([signal[x] for signal in self._signals.values()])
        else:
            raise RuntimeError('No underlying "Signal" defined!')

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

        # TODO: Handle 2D array `value`.
        for signal in self._signals.values():
            signal[x] = y

    def __contains__(self, x):
        """
        Returns whether the multi-continuous signal contains given independent
        domain :math:`x` variable.

        Parameters
        ----------
        x : numeric, array_like or slice
            Independent domain :math:`x` variable.

        Returns
        -------
        bool
            Is :math:`x` domain value contained.
        """

        if self._signals:
            return x in first_item(self._signals.values())
        else:
            raise RuntimeError('No underlying "Signal" defined!')

    def __eq__(self, other):
        """
        Returns whether the multi-continuous signal is equal to given other
        object.

        Parameters
        ----------
        other : object
            Object to test whether it is equal to multi-continuous signal.

        Returns
        -------
        bool
            Is given object equal to the multi-continuous signal.
        """

        if isinstance(other, MultiSignal):
            if all([
                    np.array_equal(self.domain, other.domain),
                    np.array_equal(self.range, other.range),
                    self.interpolator is other.interpolator,
                    self.interpolator_args == other.interpolator_args,
                    self.extrapolator is other.extrapolator,
                    self.extrapolator_args == other.extrapolator_args
            ]):
                return True

        return False

    def __neq__(self, other):
        """
        Returns whether the multi-continuous signal is not equal to given other
        object.

        Parameters
        ----------
        other : object
            Object to test whether it is not equal to the multi-continuous
            signal.

        Returns
        -------
        bool
            Is given object not equal to the multi-continuous signal.
        """

        return not (self == other)

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
        MultiSignal
            Multi-continuous signal.
        """

        multi_signal = self if in_place else self.copy()

        if isinstance(a, MultiSignal):
            assert len(self.signals) == len(a.signals), (
                '"MultiSignal" operands must have same count of '
                'underlying "Signal" components!')
            for signal_a, signal_b in zip(multi_signal.signals.values(),
                                          a.signals.values()):
                signal_a.arithmetical_operation(signal_b, operator, True)
        else:
            for signal in multi_signal.signals.values():
                signal.arithmetical_operation(a, operator, True)

        return multi_signal

    @staticmethod
    def multi_signal_unpack_data(data=None, domain=None, labels=None):
        """
        Unpack given data for multi-continuous signal instantiation.

        Parameters
        ----------
        data : Series or Dataframe or Signal or MultiSignal or array_like or \
dict_like, optional
            Data to unpack for multi-continuous signal instantiation.
        domain : array_like, optional
            Values to initialise the multiple :class:`Signal` class instances
            :attr:`Signal.domain` attribute with. If both `data` and `domain`
            arguments are defined, the latter with be used to initialise the
            :attr:`Signal.domain` attribute.

        Returns
        -------
        tuple
            Independent domain :math:`x` variable and corresponding range
            :math:`y` variable unpacked for multi-continuous signal
            instantiation.
        """

        domain_upk, range_upk, signals = None, None, None
        signals = OrderedDict()
        # TODO: Implement support for Signal class passing.
        if isinstance(data, MultiSignal):
            signals = data.signals
        elif (issubclass(type(data), Sequence) or
              isinstance(data, (tuple, list, np.ndarray, Iterator))):
            data = tsplit(list(data) if isinstance(data, Iterator) else data)
            assert data.ndim in (1, 2), (
                'User "data" must be a 1d or 2d array-like variable!')
            if data.ndim == 1:
                signals[0] = Signal(data)
            else:
                domain_upk, range_upk = ((data[0], data[1:])
                                         if domain is None else (domain, data))
                for i, range_upk_c in enumerate(range_upk):
                    signals[i] = Signal(range_upk_c, domain_upk)
        elif (issubclass(type(data), Mapping) or
              isinstance(data, (dict, OrderedDict))):
            domain_upk, range_upk = tsplit(sorted(data.items()))
            for i, range_upk in enumerate(tsplit(range_upk)):
                signals[i] = Signal(range_upk, domain_upk)
        elif is_pandas_installed():
            from pandas import DataFrame, Series

            if isinstance(data, Series):
                signals[0] = Signal(data)
            elif isinstance(data, DataFrame):
                # Check order consistency.
                domain_upk = data.index.values
                signals = OrderedDict(((label, Signal(
                    data[label], domain_upk, name=label)) for label in data))

        if domain is not None and signals is not None:
            for signal in signals.values():
                assert len(domain) == len(signal.domain), (
                    'User "domain" is not compatible with unpacked signals!')
                signal.domain = domain

        if labels is not None and signals is not None:
            assert len(labels) == len(signals), (
                'User "labels" is not compatible with unpacked signals!')
            signals = OrderedDict(
                [(labels[i], signal)
                 for i, (_key, signal) in enumerate(signals.items())])

        return signals

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
            NaNs filled multi-continuous signal.
        """

        for signal in self._signals.values():
            signal.fill_nan(method, default)

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

        if self._signals:
            return first_item(self._signals.values()).uncertainty(a)
