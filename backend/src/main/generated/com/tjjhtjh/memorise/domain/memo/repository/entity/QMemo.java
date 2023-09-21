package com.tjjhtjh.memorise.domain.memo.repository.entity;

import static com.querydsl.core.types.PathMetadataFactory.*;

import com.querydsl.core.types.dsl.*;

import com.querydsl.core.types.PathMetadata;
import javax.annotation.processing.Generated;
import com.querydsl.core.types.Path;
import com.querydsl.core.types.dsl.PathInits;


/**
 * QMemo is a Querydsl query type for Memo
 */
@Generated("com.querydsl.codegen.DefaultEntitySerializer")
public class QMemo extends EntityPathBase<Memo> {

    private static final long serialVersionUID = -514988071L;

    private static final PathInits INITS = PathInits.DIRECT2;

    public static final QMemo memo = new QMemo("memo");

    public final com.tjjhtjh.memorise.global.entity.QBaseEntity _super = new com.tjjhtjh.memorise.global.entity.QBaseEntity(this);

    public final EnumPath<AccessType> accessType = createEnum("accessType", AccessType.class);

    public final StringPath content = createString("content");

    //inherited
    public final DateTimePath<java.time.LocalDateTime> createdAt = _super.createdAt;

    public final DateTimePath<java.time.LocalDateTime> deletedAt = createDateTime("deletedAt", java.time.LocalDateTime.class);

    public final NumberPath<Integer> isDeleted = createNumber("isDeleted", Integer.class);

    public final NumberPath<Long> itemSeq = createNumber("itemSeq", Long.class);

    public final NumberPath<Long> memoSeq = createNumber("memoSeq", Long.class);

    //inherited
    public final DateTimePath<java.time.LocalDateTime> updatedAt = _super.updatedAt;

    public final com.tjjhtjh.memorise.domain.user.repository.entity.QUser user;

    public QMemo(String variable) {
        this(Memo.class, forVariable(variable), INITS);
    }

    public QMemo(Path<? extends Memo> path) {
        this(path.getType(), path.getMetadata(), PathInits.getFor(path.getMetadata(), INITS));
    }

    public QMemo(PathMetadata metadata) {
        this(metadata, PathInits.getFor(metadata, INITS));
    }

    public QMemo(PathMetadata metadata, PathInits inits) {
        this(Memo.class, metadata, inits);
    }

    public QMemo(Class<? extends Memo> type, PathMetadata metadata, PathInits inits) {
        super(type, metadata, inits);
        this.user = inits.isInitialized("user") ? new com.tjjhtjh.memorise.domain.user.repository.entity.QUser(forProperty("user")) : null;
    }

}

