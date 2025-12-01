import math
from dataclasses import dataclass
from enum import IntEnum
from string import Template
from typing import Iterator, List, Tuple

import openmdao.api as om
from openmdao.utils.om_warnings import issue_warning

import aviary.api as av


class WingType(IntEnum):
    """
    Enum class used to define wing types.

    Attributes
    ----------
    WING : int
        The main wing.
    HORIZONTAL_TAIL : int
        The rear horizontal tail wing.
    """

    WING = 0
    HORIZONTAL_TAIL = 1

    def __str__(self):
        return self.name


class Axis(IntEnum):
    """
    Enum class used to define which of the 3d axes.

    Attributes
    ----------
    X : int
        X axis.
    Y : int
        Y axis.
    Z : int
        Z axis.
    """

    X = 0
    Y = 1
    Z = 2


@dataclass
class Point3D:
    """
    A class to represent a point in 3D.

    Attributes
    ----------
    X : int
        X axis.
    Y : int
        Y axis.
    Z : int
        Z axis.
    """

    x: float = 0.0
    y: float = 0.0
    z: float = 0.0

    def translated_copy(self, dx: float, dy: float, dz: float) -> 'Point3D':
        """Return a new Point3D that is a translation of this point by (dx, dy, dz)."""
        return Point3D(self.x + dx, self.y + dy, self.z + dz)

    def reflected_copy(self, axis: Axis) -> 'Point3D':
        """Return a new Point3D that is a reflection of this point across one of the 3d axes."""
        x = self.x * (-1 if axis == Axis.X else 1)
        y = self.y * (-1 if axis == Axis.Y else 1)
        z = self.z * (-1 if axis == Axis.Z else 1)
        return Point3D(x, y, z)

    def __str__(self) -> str:
        """Return a string representation of the point in the format 'x y z'."""
        return f'{self.x} {self.y} {self.z}'


@dataclass
class Quad3D:
    """
    A class to represent a quadrilateral in 3D.

    Attributes
    ----------
    vertices : List
        A list of the Point3Ds that define the quadrilateral.
    """

    vertices: List[Point3D]

    def __post_init__(self):
        if len(self.vertices) != 4:
            raise ValueError('A quadrilateral must have exactly four vertices.')

    def translated_copy(self, dx: float, dy: float, dz: float) -> 'Quad3D':
        """Return a new Quad3D that is a translation of this quad by (dx, dy, dz)."""
        translated_vertices = [vertex.translated_copy(dx, dy, dz) for vertex in self.vertices]
        return Quad3D(translated_vertices)

    def reflected_copy(self, axis: Axis) -> 'Quad3D':
        """Return a new Quad3D that is a reflection of this quad across one of the 3 axes."""
        reflected_vertices = [vertex.reflected_copy(axis) for vertex in self.vertices]
        return Quad3D(reflected_vertices)

    def edges(self) -> Iterator[Tuple[Point3D, Point3D]]:
        """Yield each pair of adjacent vertices as edges."""
        for i in range(4):
            yield self.vertices[i], self.vertices[(i + 1) % 4]


def complete_solid(quad1, quad2):
    """
    Given two sides of a wing of some kind, defined by a quad,
    create additional quads to complete a solid/hexahedron.

    Parameters
    ----------
    quad1 : Quad3D
        Quadrilateral on one side of the wing.
    quad2 : Quad3D
        Quadrilateral on the other side of the wing.

    Returns
    -------
    hexahedron
        A list of Quad3Ds that define the six sides of the solid/hexadedron.

    """
    quads_to_complete_solid = []
    # loop over each of the 4 edges of the paired quads making new quads
    #   to complete the solid defined by the sides
    for edge1, edge2 in zip(quad1.edges(), quad2.edges()):
        # edges are defined by two vertices
        quads_to_complete_solid.append(
            Quad3D(
                [
                    edge1[0],
                    edge1[1],
                    edge2[1],
                    edge2[0],
                ]
            )
        )

    return quads_to_complete_solid


def quad3d_to_triangle_entities(quad):
    """
    Given a quadrilateral object, generate the HTML that defines the
    triangles that generate that quad.

    Parameters
    ----------
    quad : Quad3D
        Quadrilateral input.

    Returns
    -------
    html_text
        The HTML that defines the quad using two a-triangles.
    """
    vertices = quad.vertices
    entities = f"""
            <a-triangle color="white" 
            vertex-a="{vertices[0]}" 
            vertex-b="{vertices[1]}" 
            vertex-c="{vertices[2]}" 
            material="side: double"></a-triangle>

            <a-triangle color="white" 
            vertex-a="{vertices[2]}" 
            vertex-b="{vertices[3]}" 
            vertex-c="{vertices[0]}" 
            material="side: double"></a-triangle>
    """

    return entities


class AircraftModelReaderError(Exception):
    """
    Exception thrown if there was an error trying to get data from the case recorder.

    Parameters
    ----------
    msg : str
        The message string.

    Attributes
    ----------
    msg : str
        The message string.
    """

    def __init__(self, msg):
        super().__init__(msg)
        self.msg = msg


class AircraftModelReader(object):
    """
    Class used to read a case recorder file and provide methods to get Aviary variables.

    Parameters
    ----------
    case_recorder_file : str
        Path to the case recorder file.

    Attributes
    ----------
    _case_recorder_file : str
        Path to the case recorder file.
    _cr : CaseReader
        CaseReader object.
    _problem_metadata : dict
        Metadata about the problem, including the system hierarchy and connections.
    _final_case : str
        Final Problem case from the case recorder file.
    """

    def __init__(self, case_recorder_file):
        self._case_recorder_file = case_recorder_file
        self._cr = None
        self._problem_metadata = None
        self._final_case = None

    def read_case_recorder_file(self):
        """
        Read the given case recorder file.

        Parameters
        ----------
        case_recorder_file : str
            Path to the case recorder file.
        """
        cr = om.CaseReader(self._case_recorder_file)
        self._cr = cr
        self._problem_metadata = cr.problem_metadata

        model_options = cr.list_model_options(out_stream=None)
        try:
            self.aviary_options = model_options['root']['aviary_options']
        except KeyError:
            issue_warning(
                f'The case recorder file {self._case_recorder_file} does not have any metadata for the root system'
            )
            self.aviary_options = av.AviaryValues()

            # <class 'aviary.utils.aviary_values.AviaryValues'>

        if 'final' not in cr.list_cases():
            raise AircraftModelReaderError(
                f"Case recorder file, {self._case_recorder_file} does not have expected case named 'final'"
            )

        self._final_case = cr.get_case('final')

    def _write_input_output_variables(self):
        """Write out the input and output variables in the final case. For debugging."""
        self._final_case.list_inputs(
            val=True,
            return_format='list',
            prom_name=True,
            hierarchical=False,
        )
        self._final_case.list_outputs(
            val=True,
            return_format='list',
            prom_name=True,
            hierarchical=False,
        )

    def get_variable_from_case(self, var_prom_name, units=None):
        """
        Get the value of a variable from the final case.

        Parameters
        ----------
        var_prom_name : str
            Promoted name of the variable.
        units : str
            Optional. Desired units of the value.

        Returns
        -------
        value
            Value of the variable.
        """
        try:
            val = self._final_case.get_val(var_prom_name, units=units)
            return float(val)
        except KeyError:
            pass

        abs2prom = self._problem_metadata['abs2prom']
        for abs_name, prom_name in abs2prom['input'].items():
            # the phrase "_OVERRIDE" in a variable indicates it is a calculated value that we are discarding
            if prom_name == var_prom_name and '_OVERRIDE' not in abs_name:
                val = self._final_case.get_val(abs_name, units=units)
                return float(val)

        raise AircraftModelReaderError(f'Promoted name {var_prom_name} not found in final case')

    def get_variable_from_aviary_options(self, var_prom_name):
        """
        Get the value of a variable from the aviary options dict.

        Parameters
        ----------
        var_prom_name : str
            Promoted name of the variable.

        Returns
        -------
        value
            Value of the variable.
        """
        item = self.aviary_options.get_item(var_prom_name)
        if item is None:
            raise AircraftModelReaderError(
                f'Promoted name {var_prom_name} not found in aviary_options'
            )
        value, _units = item
        return value


class Fuselage(object):
    """
    Class used to represent the fuselage of the aircraft.

    Parameters
    ----------
    reader : AircraftModelReader
        The AircraftModelReader object used to read aviary variable values.

    Attributes
    ----------
    _reader : AircraftModelReader
        The AircraftModelReader object used to read aviary variable values.
    _length : float
        Length of the fuselage.
    _radius : float
        Radius of the fuselage.
    """

    def __init__(self, reader):
        self._reader = reader
        self._length = None
        self._radius = None

    def read_variables(self):
        try:
            self._length = self._reader.get_variable_from_case(
                'aircraft:fuselage:length', units='ft'
            )
            self._radius = (
                self._reader.get_variable_from_case('aircraft:fuselage:avg_diameter', units='ft')
                / 2.0
            )
        except AircraftModelReaderError as e:
            print(f'Warning: Unable to read fuselage variables due to the error: {e} ')
            raise

    def get_aframe_markup(self):
        return f"""
            <!-- fuselage -->
            <a-cylinder id="cylinder" position="0 0 0" radius="{self._radius}" height="{self._length}" 
                rotation="0 90 0" color="white"></a-cylinder>
            <!-- front cone -->
            <a-sphere color="white" radius="{self._radius}" position="0 {self._length / 2.0} 0"></a-sphere>
        """

    @property
    def length(self):
        return self._length

    @property
    def radius(self):
        return self._radius


class VerticalTail(object):
    """
    Class used to represent the vertical tail of the aircraft.

    Parameters
    ----------
    reader : AircraftModelReader
        The AircraftModelReader object used to read aviary variable values.
    fuselage : Fuselage
        The Fuselage object used to represent the fuselage of the aircraft.

    Attributes
    ----------
    _reader : AircraftModelReader
        The AircraftModelReader object used to read aviary variable values.
    _span : float
        Span of the vertical tail.
    _chord : float
        Chord of the vertical tail.
    _taper_ratio : float
        Taper ratio of the vertical tail.
    _thickness : float
        Thickness of the vertical tail.
    _fuselage : Fuselage
        The Fuselage object used to represent the fuselage of the aircraft.
    """

    def __init__(self, reader, fuselage):
        self._reader = reader
        self._span = None
        self._chord = None
        self._taper_ratio = None
        self._thickness = None
        self._fuselage = fuselage

    def read_variables(self):
        """Read the variables from the final case that are needed to define the vertical tail."""
        try:
            thickness_to_chord = self._reader.get_variable_from_case(
                'aircraft:vertical_tail:thickness_to_chord'
            )
            area = self._reader.get_variable_from_case('aircraft:vertical_tail:area', units='ft**2')
            aspect_ratio = self._reader.get_variable_from_case(
                'aircraft:vertical_tail:aspect_ratio'
            )
            self._taper_ratio = self._reader.get_variable_from_case(
                'aircraft:vertical_tail:taper_ratio'
            )
            # Calculate the span (b) using the formula b = sqrt(AR * S)
            self._span = (aspect_ratio * area) ** 0.5
            # Calculate the chord (c) using the formula c = S / b
            self._chord = area / self._span
            self._thickness = thickness_to_chord * self._chord
        except AircraftModelReaderError as e:
            print(f'Warning: Unable to read vertical tail variables due to the error: {e} ')
            raise

    def get_aframe_markup(self):
        """
        Get the A-Frame markup string.

        Returns
        -------
        str
            A-Frame markup defining the vertical tail.
        """
        # the quad that defines the left side of the vertical tail
        left_quad = Quad3D(
            [
                # point 0 - trailing edge at root
                Point3D(
                    self._fuselage.radius,
                    -self._fuselage.length / 2.0,
                    self._thickness / 2,
                ),
                # point 1 - leading edge at root
                Point3D(
                    self._fuselage.radius,
                    -self._fuselage.length / 2.0 + self._chord,
                    self._thickness / 2,
                ),
                # point 2 - leading edge at tip
                Point3D(
                    self._span + self._fuselage.radius,
                    -self._fuselage.length / 2.0 + self._chord * self._taper_ratio,
                    self._thickness / 2,
                ),
                # point 3 - trailing edge at tip
                Point3D(
                    self._span + self._fuselage.radius,
                    -self._fuselage.length / 2.0,
                    self._thickness / 2,
                ),
            ]
        )
        right_quad = left_quad.translated_copy(0, 0, -self._thickness)
        quads_to_complete_solid = complete_solid(left_quad, right_quad)

        entities = ''
        entities += f"""
        <!-- vertical tail -->
        {quad3d_to_triangle_entities(left_quad)}
        {quad3d_to_triangle_entities(right_quad)}
        """

        for quad in quads_to_complete_solid:
            entities += f'{quad3d_to_triangle_entities(quad)}'
        return entities


class HorizontalWing(object):
    """
    Class used to represent a horizontal (main or tail) wing of the aircraft.

    Parameters
    ----------
    reader : AircraftModelReader
        The AircraftModelReader object used to read aviary variable values.
    fuselage : Fuselage
        The Fuselage object used to represent the fuselage of the aircraft.
    wing_type : WingType
        Enum indicating whether this is the main wing or the horizontal tail wing.

    Attributes
    ----------
    _reader : AircraftModelReader
        The AircraftModelReader object used to read aviary variable values.
    _vertical_position : float
        Position of the wing vertically on the fuselage.
    _thickness : float
        Thickness of the wing.
    _chord : float
        Chord of the wing.
    _span : float
        Span of the wing.
    _sweep_angle : float
        Sweep angle of the wing.
    _chord_tip : float
        Chord at the top of the wing.
    _position_along_fuselage : float
        Position of the wing along the fuselage.
    _mount_location : float
        wing location on fuselage (0 = low wing, 1 = high wing, can be fractions).
    _fuselage : Fuselage
        The Fuselage object used to represent the fuselage of the aircraft.
    _wing_type : WingType
        Enum indicating whether this is the main wing or the horizontal tail wing.
    """

    def __init__(self, reader, fuselage, wing_type):
        self._reader = reader
        self._vertical_position = None
        self._thickness = None
        self._chord = None
        self._span = None
        self._sweep_angle = None
        self._chord_tip = None
        self._position_along_fuselage = None
        self._mount_location = None
        self._fuselage = fuselage
        self._wing_type = wing_type

    def read_variables(self):
        """Read the variables from the final case that are needed to define the horizontal wing."""
        if self._wing_type == WingType.WING:
            wing_type_name = 'wing'
        elif self._wing_type == WingType.HORIZONTAL_TAIL:
            wing_type_name = 'horizontal_tail'
        try:
            aspect_ratio = self._reader.get_variable_from_case(
                f'aircraft:{wing_type_name}:aspect_ratio'
            )
            taper_ratio = self._reader.get_variable_from_case(
                f'aircraft:{wing_type_name}:taper_ratio'
            )
            area = self._reader.get_variable_from_case(
                f'aircraft:{wing_type_name}:area', units='ft**2'
            )
            try:
                thickness_to_chord = self._reader.get_variable_from_case(
                    f'aircraft:{wing_type_name}:thickness_to_chord'
                )
            except AircraftModelReaderError:  # try this method if the first doesn't work
                thickness_to_chord_root = self._reader.get_variable_from_case(
                    f'aircraft:{wing_type_name}:thickness_to_chord_root'
                )
                thickness_to_chord_tip = self._reader.get_variable_from_case(
                    f'aircraft:{wing_type_name}:thickness_to_chord_tip'
                )
                thickness_to_chord = (thickness_to_chord_root + thickness_to_chord_tip) / 2.0

            self._span = (aspect_ratio * area) ** 0.5
            self._chord = area / self._span
            self._thickness = thickness_to_chord * self._chord
            self._sweep_angle = self._reader.get_variable_from_case(av.Aircraft.Wing.SWEEP)
            self._chord_tip = self._chord * taper_ratio
            if self._wing_type == WingType.WING:
                try:
                    mount_location = self._reader.get_variable_from_case(
                        'aircraft:wing:mount_location'
                    )
                except AircraftModelReaderError:
                    mount_location = 0.0
                self._vertical_position = 2.0 * (mount_location - 0.5) * self._fuselage.radius
                self._position_along_fuselage = 0.0
            elif self._wing_type == WingType.HORIZONTAL_TAIL:
                self._vertical_position = 0.0
                self._position_along_fuselage = -self._fuselage.length / 2.0 + self._chord / 2.0
        except AircraftModelReaderError as e:
            print(
                f"Warning: Unable to read horizontal wing of type '{self._wing_type}' variables due to the error: {e} "
            )
            raise

    def get_aframe_markup(self):
        """
        Get the A-Frame markup string.

        Returns
        -------
        str
            A-Frame markup defining the horizontal wing.
        """
        sweep_angle_tan = math.tan(math.radians(self._sweep_angle))

        entities = ''

        # the quad that defines the left side of the horizontal wing
        left_top_quad = Quad3D(
            [
                # point 0 - leading edge on centerline
                Point3D(
                    self._vertical_position + self._thickness / 2,
                    self._position_along_fuselage + self._chord / 2.0,
                    0.0,
                ),
                # point 1 - leading edge at tip
                Point3D(
                    self._vertical_position + self._thickness / 2,
                    self._position_along_fuselage
                    + self._chord / 2.0
                    - sweep_angle_tan * self._span / 2.0,
                    self._span / 2.0,
                ),
                # point 2 - trailing edge at tip
                Point3D(
                    self._vertical_position + self._thickness / 2,
                    self._position_along_fuselage
                    + self._chord / 2.0
                    - sweep_angle_tan * self._span / 2.0
                    - self._chord_tip,
                    self._span / 2.0,
                ),
                # point 3 - trailing edge on centerline
                Point3D(
                    self._vertical_position + self._thickness / 2,
                    self._position_along_fuselage - self._chord / 2.0,
                    0.0,
                ),
            ]
        )
        left_bottom_quad = left_top_quad.translated_copy(-self._thickness, 0, 0)
        left_quads_to_complete_solid = complete_solid(left_top_quad, left_bottom_quad)
        entities += f"""
        <!-- vertical tail -->
        {quad3d_to_triangle_entities(left_top_quad)}
        {quad3d_to_triangle_entities(left_bottom_quad)}
        """
        for quad in left_quads_to_complete_solid:
            entities += f'{quad3d_to_triangle_entities(quad)}'

        # Now the right wing
        right_top_quad = left_top_quad.reflected_copy(Axis.Z)
        right_bottom_quad = right_top_quad.translated_copy(-self._thickness, 0, 0)
        right_quads_to_complete_solid = complete_solid(right_top_quad, right_bottom_quad)
        entities += f"""
        <!-- vertical tail -->
        {quad3d_to_triangle_entities(right_top_quad)}
        {quad3d_to_triangle_entities(right_bottom_quad)}
        """

        for quad in right_quads_to_complete_solid:
            entities += f'{quad3d_to_triangle_entities(quad)}'

        return entities

    @property
    def span(self):
        return self._span

    @property
    def position_along_fuselage(self):
        return self._position_along_fuselage

    @property
    def chord(self):
        return self._chord

    @property
    def sweep_angle(self):
        return self._sweep_angle

    @property
    def vertical_position(self):
        return self._vertical_position


class Engines(object):
    """
    Class used to represent the engines of the aircraft.

    Parameters
    ----------
    reader : AircraftModelReader
        The AircraftModelReader object used to read aviary variable values.
    fuselage : Fuselage
        Fuselage object.
    wing : HorizontalWing
        The main wing object.

    Attributes
    ----------
    _reader : AircraftModelReader
        The AircraftModelReader object used to read aviary variable values.
    _fuselage : Fuselage
        Fuselage object.
    _wing : HorizontalWing
        Main wing object.
    _num_wing_engines : int
        Number of engines.
    _engine_diameter : float
        Diameter of engines.
    _engine_length : float
        Length of engines.
    _engine_locations_on_wing : float
        Location of engine on wing. Fraction of span.
    _has_propellers : bool
        Does the engine have propellers.
    """

    def __init__(self, reader, fuselage, wing):
        self._reader = reader
        self._fuselage = fuselage
        self._wing = wing
        self._num_wing_engines = None
        self._engine_diameter = None
        self._engine_length = None
        self._engine_locations_on_wing = None
        self._has_propellers = None

    def read_variables(self):
        """Read the variables from the final case that are needed to define the engines."""
        try:
            self._num_wing_engines = self._reader.get_variable_from_aviary_options(
                av.Aircraft.Engine.NUM_WING_ENGINES
            )
            self._engine_diameter = self._reader.get_variable_from_case(
                av.Aircraft.Nacelle.AVG_DIAMETER, units='ft'
            )
            self._engine_length = self._reader.get_variable_from_case(
                av.Aircraft.Nacelle.AVG_LENGTH, units='ft'
            )
            self._engine_locations_on_wing = self._reader.get_variable_from_aviary_options(
                av.Aircraft.Engine.WING_LOCATIONS
            )
            try:
                self._has_propellers = self._reader.get_variable_from_case(
                    av.Aircraft.Engine.HAS_PROPELLERS
                )
            except AircraftModelReaderError:
                self._has_propellers = False
        except AircraftModelReaderError as e:
            print(f'Warning: Unable to read engine variables due to the error: {e} ')
            raise

    def get_aframe_markup(self):
        """
        Get the A-Frame markup string.

        Returns
        -------
        str
            A-Frame markup defining the engines.
        """
        wing_span = self._wing.span
        entities = ''

        if self._engine_locations_on_wing:
            for engine_location in self._engine_locations_on_wing:
                distance_from_fuselage = engine_location * wing_span / 2.0
                distance_along_fuselage = (
                    self._wing.position_along_fuselage
                    + self._wing.chord / 2.0
                    - distance_from_fuselage * math.tan(math.radians(self._wing.sweep_angle))
                    + self._engine_length / 2.0
                )
                distance_above_fuselage = self._wing.vertical_position - self._engine_diameter / 2.0
                entities += f"""
                        <!-- engine -->
                        <a-cylinder id="cylinder" position="{distance_above_fuselage} {distance_along_fuselage} {distance_from_fuselage}" radius="{self._engine_diameter / 2}" height="{self._engine_length}" 
                            rotation="0 90 0" color="white"></a-cylinder>
                        <a-cylinder id="cylinder" position="{distance_above_fuselage} {distance_along_fuselage} {-distance_from_fuselage}" radius="{self._engine_diameter / 2}" height="{self._engine_length}" 
                            rotation="0 90 0" color="white"></a-cylinder>
                """
                if self._has_propellers:
                    propeller_blade_radius = (
                        self._engine_diameter / 10.0
                    )  # arbitrary fraction of engine diameter
                    propeller_blade_length = self._engine_diameter * 2.0
                    entities += f"""
                            <!-- engine -->
                            <a-cylinder id="cylinder" position="{distance_above_fuselage} {distance_along_fuselage + self._engine_length / 2 + propeller_blade_radius} {distance_from_fuselage}" radius="{propeller_blade_radius}" height="{propeller_blade_length}" 
                                rotation="90 135 0" color="grey"></a-cylinder>
                            <a-cylinder id="cylinder" position="{distance_above_fuselage} {distance_along_fuselage + self._engine_length / 2 + propeller_blade_radius} {distance_from_fuselage}" radius="{propeller_blade_radius}" height="{propeller_blade_length}" 
                                rotation="90 45 0" color="grey"></a-cylinder>
                            <!-- engine -->
                            <a-cylinder id="cylinder" position="{distance_above_fuselage} {distance_along_fuselage + self._engine_length / 2 + propeller_blade_radius} {-distance_from_fuselage}" radius="{propeller_blade_radius}" height="{propeller_blade_length}" 
                                rotation="90 135 0" color="grey"></a-cylinder>
                            <a-cylinder id="cylinder" position="{distance_above_fuselage} {distance_along_fuselage + self._engine_length / 2 + propeller_blade_radius} {-distance_from_fuselage}" radius="{propeller_blade_radius}" height="{propeller_blade_length}" 
                                rotation="90 45 0" color="grey"></a-cylinder>
                    """

        return entities


class Aircraft3DModel(object):
    """
    Class used to represent the 3D model of the aircraft. The A-Frame library
    is used to draw the 3D model in the Web page.

    Parameters
    ----------
    case_recorder_file : str
        Path to the case recorder file.

    Attributes
    ----------
    _reader : AircraftModelReader
        The AircraftModelReader object used to read aviary variable values.
    _entities : str
        HTML representing all the entities defining the aircraft 3D model.
    _camera_entity : str
        HTML representing all the camera in the scene.
    """

    def __init__(self, case_recorder_file):
        self._reader = AircraftModelReader(case_recorder_file)
        self._reader.read_case_recorder_file()

        # Used for debugging. Uncomment to print out the input and output variables
        #   in the final case
        # self.model_reader._write_input_output_variables()

        self._entities = ''
        self._camera_entity = ''

    def read_variables(self):
        self.fuselage = Fuselage(self._reader)
        self.fuselage.read_variables()
        self.wing = HorizontalWing(self._reader, self.fuselage, WingType.WING)
        self.wing.read_variables()
        self.engines = Engines(self._reader, self.fuselage, self.wing)
        self.engines.read_variables()
        self.horizontal_tail = HorizontalWing(self._reader, self.fuselage, WingType.HORIZONTAL_TAIL)
        self.horizontal_tail.read_variables()
        self.vertical_tail = VerticalTail(self._reader, self.fuselage)
        self.vertical_tail.read_variables()

    def get_aframe_markup(self):
        self._entities += self.fuselage.get_aframe_markup()
        self._entities += self.wing.get_aframe_markup()
        self._entities += self.engines.get_aframe_markup()
        self._entities += self.horizontal_tail.get_aframe_markup()
        self._entities += self.vertical_tail.get_aframe_markup()

    def get_camera_entity(self, fuselage_length):
        y_camera = fuselage_length / 2.0
        z_camera = fuselage_length
        self._camera_entity = f"""
        <a-entity camera look-controls="enabled: false" orbit-controls="target: 0 0 0; 
            minDistance: 2; maxDistance: 180; initialPosition: 0 {y_camera} {z_camera}; rotateSpeed: 0.5"></a-entity>
        """

    def write_file(self, aircraft_3d_template_filepath, outfilepath):
        with open(aircraft_3d_template_filepath, 'r', encoding='utf-8') as f:
            aircraft_3d_template = f.read()

        with open(outfilepath, 'w') as f:
            template = Template(aircraft_3d_template)

            s = template.substitute(entities=self._entities, camera_entity=self._camera_entity)
            f.write(s)
