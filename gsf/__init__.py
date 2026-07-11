# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2018--, Aura Obreja and the GalacticStructureFinder (gsf) contributors.

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

from importlib.metadata import version as _version, PackageNotFoundError
try:
    __version__ = _version("galactic-structure-finder")
except PackageNotFoundError:      # running from a source tree without install
    __version__ = "unknown"

from .gsf import gsf, gsf_loop
from .donaming import tag_components
