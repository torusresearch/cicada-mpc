# Copyright 2021 National Technology & Engineering Solutions
# of Sandia, LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS,
# the U.S. Government retains certain rights in this software.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import test

from behave import *
import numpy

import cicada.arithmetic


@given(u'a Field with default order')
def step_impl(context):
    if "fields" not in context:
        context.fields = []
    context.fields.append(cicada.arithmetic.Field())


@given(u'a Field with order {order}')
def step_impl(context, order):
    order = eval(order)
    if "fields" not in context:
        context.fields = []
    context.fields.append(cicada.arithmetic.Field(order=order))


@given(u'a field array {x}')
def step_impl(context, x):
    x = numpy.array(eval(x))
    field = context.fields[-1]
    if "fieldarrays" not in context:
        context.fieldarrays = []
    context.fieldarrays.append(field(x))


@when(u'generating a field array of zeros with shape {shape}')
def step_impl(context, shape):
    shape = eval(shape)
    field = context.fields[-1]
    if "fieldarrays" not in context:
        context.fieldarrays = []
    context.fieldarrays.append(field.zeros(shape))


@when(u'generating a field array of zeros like {other}')
def step_impl(context, other):
    other = numpy.array(eval(other))
    field = context.fields[-1]
    if "fieldarrays" not in context:
        context.fieldarrays = []
    context.fieldarrays.append(field.zeros_like(other))


@when(u'the field array is negated')
def step_impl(context):
    field = context.fields[-1]
    fieldarray = context.fieldarrays.pop()
    context.fieldarrays.append(field.negative(fieldarray))


@when(u'the field array is summed')
def step_impl(context):
    field = context.fields[-1]
    fieldarray = context.fieldarrays.pop()
    context.fieldarrays.append(field.sum(fieldarray))


@when(u'the second field array is subtracted from the first')
def step_impl(context):
    field = context.fields[-1]
    b = context.fieldarrays.pop()
    a = context.fieldarrays.pop()
    context.fieldarrays.append(field.subtract(a, b))


@then(u'the field array should match {result}')
def step_impl(context, result):
    field = context.fields[-1]
    result = field(eval(result))
    fieldarray = context.fieldarrays.pop()
    numpy.testing.assert_array_equal(fieldarray, result)


@then(u'the fields should compare equal')
def step_impl(context):
    lhs, rhs = context.fields
    test.assert_true(lhs == rhs)


@then(u'the fields should compare unequal')
def step_impl(context):
    lhs, rhs = context.fields
    test.assert_true(lhs != rhs)


