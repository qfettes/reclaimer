/**
 * Autogenerated by Thrift Compiler (0.12.0)
 *
 * DO NOT EDIT UNLESS YOU ARE SURE THAT YOU KNOW WHAT YOU ARE DOING
 *  @generated
 */
#include "MediaFilterService.h"

namespace social_network {


MediaFilterService_UploadMedia_args::~MediaFilterService_UploadMedia_args() throw() {
}


uint32_t MediaFilterService_UploadMedia_args::read(::apache::thrift::protocol::TProtocol* iprot) {

  ::apache::thrift::protocol::TInputRecursionTracker tracker(*iprot);
  uint32_t xfer = 0;
  std::string fname;
  ::apache::thrift::protocol::TType ftype;
  int16_t fid;

  xfer += iprot->readStructBegin(fname);

  using ::apache::thrift::protocol::TProtocolException;


  while (true)
  {
    xfer += iprot->readFieldBegin(fname, ftype, fid);
    if (ftype == ::apache::thrift::protocol::T_STOP) {
      break;
    }
    switch (fid)
    {
      case 1:
        if (ftype == ::apache::thrift::protocol::T_I64) {
          xfer += iprot->readI64(this->req_id);
          this->__isset.req_id = true;
        } else {
          xfer += iprot->skip(ftype);
        }
        break;
      case 2:
        if (ftype == ::apache::thrift::protocol::T_LIST) {
          {
            this->media_types.clear();
            uint32_t _size455;
            ::apache::thrift::protocol::TType _etype458;
            xfer += iprot->readListBegin(_etype458, _size455);
            this->media_types.resize(_size455);
            uint32_t _i459;
            for (_i459 = 0; _i459 < _size455; ++_i459)
            {
              xfer += iprot->readString(this->media_types[_i459]);
            }
            xfer += iprot->readListEnd();
          }
          this->__isset.media_types = true;
        } else {
          xfer += iprot->skip(ftype);
        }
        break;
      case 3:
        if (ftype == ::apache::thrift::protocol::T_LIST) {
          {
            this->medium.clear();
            uint32_t _size460;
            ::apache::thrift::protocol::TType _etype463;
            xfer += iprot->readListBegin(_etype463, _size460);
            this->medium.resize(_size460);
            uint32_t _i464;
            for (_i464 = 0; _i464 < _size460; ++_i464)
            {
              xfer += iprot->readString(this->medium[_i464]);
            }
            xfer += iprot->readListEnd();
          }
          this->__isset.medium = true;
        } else {
          xfer += iprot->skip(ftype);
        }
        break;
      case 4:
        if (ftype == ::apache::thrift::protocol::T_MAP) {
          {
            this->carrier.clear();
            uint32_t _size465;
            ::apache::thrift::protocol::TType _ktype466;
            ::apache::thrift::protocol::TType _vtype467;
            xfer += iprot->readMapBegin(_ktype466, _vtype467, _size465);
            uint32_t _i469;
            for (_i469 = 0; _i469 < _size465; ++_i469)
            {
              std::string _key470;
              xfer += iprot->readString(_key470);
              std::string& _val471 = this->carrier[_key470];
              xfer += iprot->readString(_val471);
            }
            xfer += iprot->readMapEnd();
          }
          this->__isset.carrier = true;
        } else {
          xfer += iprot->skip(ftype);
        }
        break;
      default:
        xfer += iprot->skip(ftype);
        break;
    }
    xfer += iprot->readFieldEnd();
  }

  xfer += iprot->readStructEnd();

  return xfer;
}

uint32_t MediaFilterService_UploadMedia_args::write(::apache::thrift::protocol::TProtocol* oprot) const {
  uint32_t xfer = 0;
  ::apache::thrift::protocol::TOutputRecursionTracker tracker(*oprot);
  xfer += oprot->writeStructBegin("MediaFilterService_UploadMedia_args");

  xfer += oprot->writeFieldBegin("req_id", ::apache::thrift::protocol::T_I64, 1);
  xfer += oprot->writeI64(this->req_id);
  xfer += oprot->writeFieldEnd();

  xfer += oprot->writeFieldBegin("media_types", ::apache::thrift::protocol::T_LIST, 2);
  {
    xfer += oprot->writeListBegin(::apache::thrift::protocol::T_STRING, static_cast<uint32_t>(this->media_types.size()));
    std::vector<std::string> ::const_iterator _iter472;
    for (_iter472 = this->media_types.begin(); _iter472 != this->media_types.end(); ++_iter472)
    {
      xfer += oprot->writeString((*_iter472));
    }
    xfer += oprot->writeListEnd();
  }
  xfer += oprot->writeFieldEnd();

  xfer += oprot->writeFieldBegin("medium", ::apache::thrift::protocol::T_LIST, 3);
  {
    xfer += oprot->writeListBegin(::apache::thrift::protocol::T_STRING, static_cast<uint32_t>(this->medium.size()));
    std::vector<std::string> ::const_iterator _iter473;
    for (_iter473 = this->medium.begin(); _iter473 != this->medium.end(); ++_iter473)
    {
      xfer += oprot->writeString((*_iter473));
    }
    xfer += oprot->writeListEnd();
  }
  xfer += oprot->writeFieldEnd();

  xfer += oprot->writeFieldBegin("carrier", ::apache::thrift::protocol::T_MAP, 4);
  {
    xfer += oprot->writeMapBegin(::apache::thrift::protocol::T_STRING, ::apache::thrift::protocol::T_STRING, static_cast<uint32_t>(this->carrier.size()));
    std::map<std::string, std::string> ::const_iterator _iter474;
    for (_iter474 = this->carrier.begin(); _iter474 != this->carrier.end(); ++_iter474)
    {
      xfer += oprot->writeString(_iter474->first);
      xfer += oprot->writeString(_iter474->second);
    }
    xfer += oprot->writeMapEnd();
  }
  xfer += oprot->writeFieldEnd();

  xfer += oprot->writeFieldStop();
  xfer += oprot->writeStructEnd();
  return xfer;
}


MediaFilterService_UploadMedia_pargs::~MediaFilterService_UploadMedia_pargs() throw() {
}


uint32_t MediaFilterService_UploadMedia_pargs::write(::apache::thrift::protocol::TProtocol* oprot) const {
  uint32_t xfer = 0;
  ::apache::thrift::protocol::TOutputRecursionTracker tracker(*oprot);
  xfer += oprot->writeStructBegin("MediaFilterService_UploadMedia_pargs");

  xfer += oprot->writeFieldBegin("req_id", ::apache::thrift::protocol::T_I64, 1);
  xfer += oprot->writeI64((*(this->req_id)));
  xfer += oprot->writeFieldEnd();

  xfer += oprot->writeFieldBegin("media_types", ::apache::thrift::protocol::T_LIST, 2);
  {
    xfer += oprot->writeListBegin(::apache::thrift::protocol::T_STRING, static_cast<uint32_t>((*(this->media_types)).size()));
    std::vector<std::string> ::const_iterator _iter475;
    for (_iter475 = (*(this->media_types)).begin(); _iter475 != (*(this->media_types)).end(); ++_iter475)
    {
      xfer += oprot->writeString((*_iter475));
    }
    xfer += oprot->writeListEnd();
  }
  xfer += oprot->writeFieldEnd();

  xfer += oprot->writeFieldBegin("medium", ::apache::thrift::protocol::T_LIST, 3);
  {
    xfer += oprot->writeListBegin(::apache::thrift::protocol::T_STRING, static_cast<uint32_t>((*(this->medium)).size()));
    std::vector<std::string> ::const_iterator _iter476;
    for (_iter476 = (*(this->medium)).begin(); _iter476 != (*(this->medium)).end(); ++_iter476)
    {
      xfer += oprot->writeString((*_iter476));
    }
    xfer += oprot->writeListEnd();
  }
  xfer += oprot->writeFieldEnd();

  xfer += oprot->writeFieldBegin("carrier", ::apache::thrift::protocol::T_MAP, 4);
  {
    xfer += oprot->writeMapBegin(::apache::thrift::protocol::T_STRING, ::apache::thrift::protocol::T_STRING, static_cast<uint32_t>((*(this->carrier)).size()));
    std::map<std::string, std::string> ::const_iterator _iter477;
    for (_iter477 = (*(this->carrier)).begin(); _iter477 != (*(this->carrier)).end(); ++_iter477)
    {
      xfer += oprot->writeString(_iter477->first);
      xfer += oprot->writeString(_iter477->second);
    }
    xfer += oprot->writeMapEnd();
  }
  xfer += oprot->writeFieldEnd();

  xfer += oprot->writeFieldStop();
  xfer += oprot->writeStructEnd();
  return xfer;
}


MediaFilterService_UploadMedia_result::~MediaFilterService_UploadMedia_result() throw() {
}


uint32_t MediaFilterService_UploadMedia_result::read(::apache::thrift::protocol::TProtocol* iprot) {

  ::apache::thrift::protocol::TInputRecursionTracker tracker(*iprot);
  uint32_t xfer = 0;
  std::string fname;
  ::apache::thrift::protocol::TType ftype;
  int16_t fid;

  xfer += iprot->readStructBegin(fname);

  using ::apache::thrift::protocol::TProtocolException;


  while (true)
  {
    xfer += iprot->readFieldBegin(fname, ftype, fid);
    if (ftype == ::apache::thrift::protocol::T_STOP) {
      break;
    }
    switch (fid)
    {
      case 0:
        if (ftype == ::apache::thrift::protocol::T_LIST) {
          {
            this->success.clear();
            uint32_t _size478;
            ::apache::thrift::protocol::TType _etype481;
            xfer += iprot->readListBegin(_etype481, _size478);
            this->success.resize(_size478);
            uint32_t _i482;
            for (_i482 = 0; _i482 < _size478; ++_i482)
            {
              xfer += iprot->readBool(this->success[_i482]);
            }
            xfer += iprot->readListEnd();
          }
          this->__isset.success = true;
        } else {
          xfer += iprot->skip(ftype);
        }
        break;
      case 1:
        if (ftype == ::apache::thrift::protocol::T_STRUCT) {
          xfer += this->se.read(iprot);
          this->__isset.se = true;
        } else {
          xfer += iprot->skip(ftype);
        }
        break;
      default:
        xfer += iprot->skip(ftype);
        break;
    }
    xfer += iprot->readFieldEnd();
  }

  xfer += iprot->readStructEnd();

  return xfer;
}

uint32_t MediaFilterService_UploadMedia_result::write(::apache::thrift::protocol::TProtocol* oprot) const {

  uint32_t xfer = 0;

  xfer += oprot->writeStructBegin("MediaFilterService_UploadMedia_result");

  if (this->__isset.success) {
    xfer += oprot->writeFieldBegin("success", ::apache::thrift::protocol::T_LIST, 0);
    {
      xfer += oprot->writeListBegin(::apache::thrift::protocol::T_BOOL, static_cast<uint32_t>(this->success.size()));
      std::vector<bool> ::const_iterator _iter483;
      for (_iter483 = this->success.begin(); _iter483 != this->success.end(); ++_iter483)
      {
        xfer += oprot->writeBool((*_iter483));
      }
      xfer += oprot->writeListEnd();
    }
    xfer += oprot->writeFieldEnd();
  } else if (this->__isset.se) {
    xfer += oprot->writeFieldBegin("se", ::apache::thrift::protocol::T_STRUCT, 1);
    xfer += this->se.write(oprot);
    xfer += oprot->writeFieldEnd();
  }
  xfer += oprot->writeFieldStop();
  xfer += oprot->writeStructEnd();
  return xfer;
}


MediaFilterService_UploadMedia_presult::~MediaFilterService_UploadMedia_presult() throw() {
}


uint32_t MediaFilterService_UploadMedia_presult::read(::apache::thrift::protocol::TProtocol* iprot) {

  ::apache::thrift::protocol::TInputRecursionTracker tracker(*iprot);
  uint32_t xfer = 0;
  std::string fname;
  ::apache::thrift::protocol::TType ftype;
  int16_t fid;

  xfer += iprot->readStructBegin(fname);

  using ::apache::thrift::protocol::TProtocolException;


  while (true)
  {
    xfer += iprot->readFieldBegin(fname, ftype, fid);
    if (ftype == ::apache::thrift::protocol::T_STOP) {
      break;
    }
    switch (fid)
    {
      case 0:
        if (ftype == ::apache::thrift::protocol::T_LIST) {
          {
            (*(this->success)).clear();
            uint32_t _size484;
            ::apache::thrift::protocol::TType _etype487;
            xfer += iprot->readListBegin(_etype487, _size484);
            (*(this->success)).resize(_size484);
            uint32_t _i488;
            for (_i488 = 0; _i488 < _size484; ++_i488)
            {
              xfer += iprot->readBool((*(this->success))[_i488]);
            }
            xfer += iprot->readListEnd();
          }
          this->__isset.success = true;
        } else {
          xfer += iprot->skip(ftype);
        }
        break;
      case 1:
        if (ftype == ::apache::thrift::protocol::T_STRUCT) {
          xfer += this->se.read(iprot);
          this->__isset.se = true;
        } else {
          xfer += iprot->skip(ftype);
        }
        break;
      default:
        xfer += iprot->skip(ftype);
        break;
    }
    xfer += iprot->readFieldEnd();
  }

  xfer += iprot->readStructEnd();

  return xfer;
}

void MediaFilterServiceClient::UploadMedia(std::vector<bool> & _return, const int64_t req_id, const std::vector<std::string> & media_types, const std::vector<std::string> & medium, const std::map<std::string, std::string> & carrier)
{
  send_UploadMedia(req_id, media_types, medium, carrier);
  recv_UploadMedia(_return);
}

void MediaFilterServiceClient::send_UploadMedia(const int64_t req_id, const std::vector<std::string> & media_types, const std::vector<std::string> & medium, const std::map<std::string, std::string> & carrier)
{
  int32_t cseqid = 0;
  oprot_->writeMessageBegin("UploadMedia", ::apache::thrift::protocol::T_CALL, cseqid);

  MediaFilterService_UploadMedia_pargs args;
  args.req_id = &req_id;
  args.media_types = &media_types;
  args.medium = &medium;
  args.carrier = &carrier;
  args.write(oprot_);

  oprot_->writeMessageEnd();
  oprot_->getTransport()->writeEnd();
  oprot_->getTransport()->flush();
}

void MediaFilterServiceClient::recv_UploadMedia(std::vector<bool> & _return)
{

  int32_t rseqid = 0;
  std::string fname;
  ::apache::thrift::protocol::TMessageType mtype;

  iprot_->readMessageBegin(fname, mtype, rseqid);
  if (mtype == ::apache::thrift::protocol::T_EXCEPTION) {
    ::apache::thrift::TApplicationException x;
    x.read(iprot_);
    iprot_->readMessageEnd();
    iprot_->getTransport()->readEnd();
    throw x;
  }
  if (mtype != ::apache::thrift::protocol::T_REPLY) {
    iprot_->skip(::apache::thrift::protocol::T_STRUCT);
    iprot_->readMessageEnd();
    iprot_->getTransport()->readEnd();
  }
  if (fname.compare("UploadMedia") != 0) {
    iprot_->skip(::apache::thrift::protocol::T_STRUCT);
    iprot_->readMessageEnd();
    iprot_->getTransport()->readEnd();
  }
  MediaFilterService_UploadMedia_presult result;
  result.success = &_return;
  result.read(iprot_);
  iprot_->readMessageEnd();
  iprot_->getTransport()->readEnd();

  if (result.__isset.success) {
    // _return pointer has now been filled
    return;
  }
  if (result.__isset.se) {
    throw result.se;
  }
  throw ::apache::thrift::TApplicationException(::apache::thrift::TApplicationException::MISSING_RESULT, "UploadMedia failed: unknown result");
}

bool MediaFilterServiceProcessor::dispatchCall(::apache::thrift::protocol::TProtocol* iprot, ::apache::thrift::protocol::TProtocol* oprot, const std::string& fname, int32_t seqid, void* callContext) {
  ProcessMap::iterator pfn;
  pfn = processMap_.find(fname);
  if (pfn == processMap_.end()) {
    iprot->skip(::apache::thrift::protocol::T_STRUCT);
    iprot->readMessageEnd();
    iprot->getTransport()->readEnd();
    ::apache::thrift::TApplicationException x(::apache::thrift::TApplicationException::UNKNOWN_METHOD, "Invalid method name: '"+fname+"'");
    oprot->writeMessageBegin(fname, ::apache::thrift::protocol::T_EXCEPTION, seqid);
    x.write(oprot);
    oprot->writeMessageEnd();
    oprot->getTransport()->writeEnd();
    oprot->getTransport()->flush();
    return true;
  }
  (this->*(pfn->second))(seqid, iprot, oprot, callContext);
  return true;
}

void MediaFilterServiceProcessor::process_UploadMedia(int32_t seqid, ::apache::thrift::protocol::TProtocol* iprot, ::apache::thrift::protocol::TProtocol* oprot, void* callContext)
{
  void* ctx = NULL;
  if (this->eventHandler_.get() != NULL) {
    ctx = this->eventHandler_->getContext("MediaFilterService.UploadMedia", callContext);
  }
  ::apache::thrift::TProcessorContextFreer freer(this->eventHandler_.get(), ctx, "MediaFilterService.UploadMedia");

  if (this->eventHandler_.get() != NULL) {
    this->eventHandler_->preRead(ctx, "MediaFilterService.UploadMedia");
  }

  MediaFilterService_UploadMedia_args args;
  args.read(iprot);
  iprot->readMessageEnd();
  uint32_t bytes = iprot->getTransport()->readEnd();

  if (this->eventHandler_.get() != NULL) {
    this->eventHandler_->postRead(ctx, "MediaFilterService.UploadMedia", bytes);
  }

  MediaFilterService_UploadMedia_result result;
  try {
    iface_->UploadMedia(result.success, args.req_id, args.media_types, args.medium, args.carrier);
    result.__isset.success = true;
  } catch (ServiceException &se) {
    result.se = se;
    result.__isset.se = true;
  } catch (const std::exception& e) {
    if (this->eventHandler_.get() != NULL) {
      this->eventHandler_->handlerError(ctx, "MediaFilterService.UploadMedia");
    }

    ::apache::thrift::TApplicationException x(e.what());
    oprot->writeMessageBegin("UploadMedia", ::apache::thrift::protocol::T_EXCEPTION, seqid);
    x.write(oprot);
    oprot->writeMessageEnd();
    oprot->getTransport()->writeEnd();
    oprot->getTransport()->flush();
    return;
  }

  if (this->eventHandler_.get() != NULL) {
    this->eventHandler_->preWrite(ctx, "MediaFilterService.UploadMedia");
  }

  oprot->writeMessageBegin("UploadMedia", ::apache::thrift::protocol::T_REPLY, seqid);
  result.write(oprot);
  oprot->writeMessageEnd();
  bytes = oprot->getTransport()->writeEnd();
  oprot->getTransport()->flush();

  if (this->eventHandler_.get() != NULL) {
    this->eventHandler_->postWrite(ctx, "MediaFilterService.UploadMedia", bytes);
  }
}

::apache::thrift::stdcxx::shared_ptr< ::apache::thrift::TProcessor > MediaFilterServiceProcessorFactory::getProcessor(const ::apache::thrift::TConnectionInfo& connInfo) {
  ::apache::thrift::ReleaseHandler< MediaFilterServiceIfFactory > cleanup(handlerFactory_);
  ::apache::thrift::stdcxx::shared_ptr< MediaFilterServiceIf > handler(handlerFactory_->getHandler(connInfo), cleanup);
  ::apache::thrift::stdcxx::shared_ptr< ::apache::thrift::TProcessor > processor(new MediaFilterServiceProcessor(handler));
  return processor;
}

void MediaFilterServiceConcurrentClient::UploadMedia(std::vector<bool> & _return, const int64_t req_id, const std::vector<std::string> & media_types, const std::vector<std::string> & medium, const std::map<std::string, std::string> & carrier)
{
  int32_t seqid = send_UploadMedia(req_id, media_types, medium, carrier);
  recv_UploadMedia(_return, seqid);
}

int32_t MediaFilterServiceConcurrentClient::send_UploadMedia(const int64_t req_id, const std::vector<std::string> & media_types, const std::vector<std::string> & medium, const std::map<std::string, std::string> & carrier)
{
  int32_t cseqid = this->sync_.generateSeqId();
  ::apache::thrift::async::TConcurrentSendSentry sentry(&this->sync_);
  oprot_->writeMessageBegin("UploadMedia", ::apache::thrift::protocol::T_CALL, cseqid);

  MediaFilterService_UploadMedia_pargs args;
  args.req_id = &req_id;
  args.media_types = &media_types;
  args.medium = &medium;
  args.carrier = &carrier;
  args.write(oprot_);

  oprot_->writeMessageEnd();
  oprot_->getTransport()->writeEnd();
  oprot_->getTransport()->flush();

  sentry.commit();
  return cseqid;
}

void MediaFilterServiceConcurrentClient::recv_UploadMedia(std::vector<bool> & _return, const int32_t seqid)
{

  int32_t rseqid = 0;
  std::string fname;
  ::apache::thrift::protocol::TMessageType mtype;

  // the read mutex gets dropped and reacquired as part of waitForWork()
  // The destructor of this sentry wakes up other clients
  ::apache::thrift::async::TConcurrentRecvSentry sentry(&this->sync_, seqid);

  while(true) {
    if(!this->sync_.getPending(fname, mtype, rseqid)) {
      iprot_->readMessageBegin(fname, mtype, rseqid);
    }
    if(seqid == rseqid) {
      if (mtype == ::apache::thrift::protocol::T_EXCEPTION) {
        ::apache::thrift::TApplicationException x;
        x.read(iprot_);
        iprot_->readMessageEnd();
        iprot_->getTransport()->readEnd();
        sentry.commit();
        throw x;
      }
      if (mtype != ::apache::thrift::protocol::T_REPLY) {
        iprot_->skip(::apache::thrift::protocol::T_STRUCT);
        iprot_->readMessageEnd();
        iprot_->getTransport()->readEnd();
      }
      if (fname.compare("UploadMedia") != 0) {
        iprot_->skip(::apache::thrift::protocol::T_STRUCT);
        iprot_->readMessageEnd();
        iprot_->getTransport()->readEnd();

        // in a bad state, don't commit
        using ::apache::thrift::protocol::TProtocolException;
        throw TProtocolException(TProtocolException::INVALID_DATA);
      }
      MediaFilterService_UploadMedia_presult result;
      result.success = &_return;
      result.read(iprot_);
      iprot_->readMessageEnd();
      iprot_->getTransport()->readEnd();

      if (result.__isset.success) {
        // _return pointer has now been filled
        sentry.commit();
        return;
      }
      if (result.__isset.se) {
        sentry.commit();
        throw result.se;
      }
      // in a bad state, don't commit
      throw ::apache::thrift::TApplicationException(::apache::thrift::TApplicationException::MISSING_RESULT, "UploadMedia failed: unknown result");
    }
    // seqid != rseqid
    this->sync_.updatePending(fname, mtype, rseqid);

    // this will temporarily unlock the readMutex, and let other clients get work done
    this->sync_.waitForWork(seqid);
  } // end while(true)
}

} // namespace

