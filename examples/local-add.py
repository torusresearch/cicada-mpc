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

import numpy

import cicada.additive
import cicada.communicator

logging.basicConfig(level=logging.INFO)

def main(communicator):
    log = cicada.Logger(logging.getLogger(), communicator)
    protocol = cicada.additive.AdditiveProtocol(communicator)

    secret = numpy.array(3) if communicator.rank == 0 else None
    log.info(f"Player {communicator.rank} secret: {secret}")

    share = protocol.share(src=0, secret=protocol.encoder.encode(secret), shape=())
    log.info(f"Player {communicator.rank} share: {share}")

    log.info(f"Player {communicator.rank} adding 2.3 to local share.", src=2)
    if communicator.rank == 2:
        protocol.encoder.inplace_add(share.storage, protocol.encoder.encode(numpy.array(2.3)))
    log.info(f"Player {communicator.rank} modified share: {share}")

    revealed = protocol.encoder.decode(protocol.reveal(share))
    log.info(f"Player {communicator.rank} revealed: {revealed}")

cicada.communicator.SocketCommunicator.run(main, world_size=3)

