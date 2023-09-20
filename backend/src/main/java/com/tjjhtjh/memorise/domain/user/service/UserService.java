package com.tjjhtjh.memorise.domain.user.service;

import com.tjjhtjh.memorise.domain.user.repository.entity.User;
import com.tjjhtjh.memorise.domain.user.service.dto.request.JoinRequest;
import com.tjjhtjh.memorise.domain.user.service.dto.response.UserInfoResponse;

public interface UserService {

    void join(JoinRequest joinRequest);
    User getUserInfo(String email);

}
