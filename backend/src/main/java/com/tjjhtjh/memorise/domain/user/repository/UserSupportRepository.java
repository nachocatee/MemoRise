package com.tjjhtjh.memorise.domain.user.repository;

import com.tjjhtjh.memorise.domain.user.service.dto.response.UserListResponse;

public interface UserSupportRepository {
    UserListResponse findByNicknameContainingAndIsDeletedFalse(String nickname);

}
