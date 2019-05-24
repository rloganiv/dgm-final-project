from squawkbox.utils import Registrable


def test_registrable():
    # Create a registrable class
    class Thing(Registrable):
        pass

    # Register a subclass
    @Thing.register('a-type')
    class ATypeThing(Thing):
        pass

    # Check that subclass can be gotten from parent's registry
    a_type = Thing.get('a-type')
    a_type_instance = a_type()
