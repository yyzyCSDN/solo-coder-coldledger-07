from datetime import timedelta
from .inventory.fefo import reserve
from .quality.release import may_release
from .transport.connections import choose
from .transport.lane_qualification import (
    BookingRejected,
    CarrierStrategy,
    LaneBookingService,
    LaneQualification,
    LaneQualificationRegistry,
    QualificationEvidence,
    QuoteRequest,
    SnapshotConflict,
    evaluate_snapshot,
)
class ColdLedgerService:
    def reserve_inventory(self,*args,**kwargs): return reserve(*args,**kwargs)
    def release_decision(self,*args,**kwargs): return may_release(*args,**kwargs)
    def next_connection(self,inbound,candidates,minutes): return choose(inbound,candidates,timedelta(minutes=minutes))
    def lane_qualification_registry(self): return LaneQualificationRegistry()
    def lane_booking_service(self,registry=None): return LaneBookingService(registry or self.lane_qualification_registry())
    def lane_quote(self,service,request,at=None): return service.quote(request,at)
    def lane_book(self,service,quote_id,at=None,on_conflict='reject'): return service.book(quote_id,at,on_conflict)
    def lane_snapshot_for_booking(self,service,booking_id): return service.snapshot_for_booking(booking_id)
    def lane_eligibility(self,snapshot,at,product_class,payload_kg=None,carrier_id=None,valid_through=None):
        return evaluate_snapshot(snapshot,at,product_class,payload_kg,carrier_id,valid_through)
    def health(self): return {'service':'coldledger','status':'ok','runtime_dependencies':0}
